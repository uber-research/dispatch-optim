import numpy as np
import pulp as plp
import itertools
import operator
import time
import logging
from dataclasses import dataclass, field
from numba import njit, jit, objmode, int64, float64
from numba.experimental import jitclass
from routing_charging.search_utils import (
      update_penalty,
      copy_solution
)

import os
from datetime import timedelta
from routing_charging.neighborhoods import apply_neigh, shake

@dataclass
class ElectricMilp():
  """ Electic version - routing and recharging MILP

      Args :
            cost: table containing $ cost to link request r1 to r2
            time: table containing time to travel from i to j
            energy: table containing energy used %SoC to travel from i to j
            requests: table containing request info : id | r- | r+ | t_r- | t_r+ | beta_r
            aircraft: table containing aircraft carac : id | eta | start_id |
            skyports: table containing skyport carac : id | landing fee |
            gamma_s: slow charge rate % SoC per minute
            gamma_f: fast charge rate % SoC per minute
            pe: price of electricity in $/kwh
            delta: number of minutes to board or unboard an evtol
            lbda_u: penalty term for unserved demands
            lbda_f: penalty term for using fast charge
            min_soc: minimum soc allowed to takeoff.
            T: time steps
  """
  cost: np.ndarray
  time: np.ndarray
  energy: np.ndarray
  requests: np.ndarray
  aircraft: np.ndarray
  skyports: np.ndarray
  gamma_s: float
  gamma_f: float
  pe: float
  delta: int
  lbda_u: float
  lbda_f: float
  min_soc: int
  T: list
  big_M: int = 5000

  def build_model(self):
    """ Build the Integers Linear Program instance.
      --------------
      Params :
              None
      --------------
      Returns :
              None

    """
    n_aircraft = self.aircraft.shape[0]
    n_skyports = self.skyports.shape[0]
    n_requests = self.requests.shape[0]
    ORIGIN = 1
    DEST = 2
    ORIGIN_TIME = 3
    DEST_TIME = 4
    aircraft_ids = list(self.aircraft[:, 0])
    skyports_ids = list(self.skyports[:, 0])
    requests_ids = list(self.requests[:, 0])

    # First creates the master problem : assigns a pair of requests to evtols
    self.y = plp.LpVariable.dicts("vrr", (aircraft_ids, requests_ids, requests_ids), 0, 1, plp.LpInteger)

    # Variable indicating slow charge
    self.sa = plp.LpVariable.dicts("srra", (aircraft_ids, requests_ids, requests_ids), 0, 1, plp.LpInteger)
    self.sb = plp.LpVariable.dicts("srrb", (aircraft_ids, requests_ids, requests_ids), 0, 1, plp.LpInteger)

    # Variable counting the amount of electrivity bought by vtols
    self.b = plp.LpVariable.dicts("brr", (aircraft_ids, requests_ids, requests_ids), 0, 100, plp.LpContinuous)
    self.ba = plp.LpVariable.dicts("brra", (aircraft_ids, requests_ids, requests_ids), 0, 100, plp.LpContinuous)
    self.bb = plp.LpVariable.dicts("brrb", (aircraft_ids, requests_ids, requests_ids), 0, 100, plp.LpContinuous)

    # Variable monitoring charge level of evtols
    self.e_before = plp.LpVariable.dicts("ebef", (aircraft_ids, requests_ids), 0, 100, plp.LpContinuous)
    self.e_after = plp.LpVariable.dicts("eaft", (aircraft_ids, requests_ids), 0, 100, plp.LpContinuous)

    # Binary variables used to linearize a min function
    self.za = plp.LpVariable.dicts("zrra", (aircraft_ids, requests_ids, requests_ids), 0, 1, plp.LpInteger)
    self.zb = plp.LpVariable.dicts("zrrb", (aircraft_ids, requests_ids, requests_ids), 0, 1, plp.LpInteger)


    # Integers variables used to linearize products
    self.pbs = plp.LpVariable.dicts("pbs", (aircraft_ids, requests_ids, requests_ids), 0, 100, plp.LpContinuous)
    self.pbf = plp.LpVariable.dicts("pbf", (aircraft_ids, requests_ids, requests_ids), 0, 100, plp.LpContinuous)
    self.pas = plp.LpVariable.dicts("pas", (aircraft_ids, requests_ids, requests_ids), 0, 100, plp.LpContinuous)
    self.paf = plp.LpVariable.dicts("paf", (aircraft_ids, requests_ids, requests_ids), 0, 100, plp.LpContinuous)


    # Time variables
    self.ta = plp.LpVariable.dicts("ta", (aircraft_ids, requests_ids, requests_ids), 0, 100, plp.LpContinuous)
    self.tb = plp.LpVariable.dicts("tb", (aircraft_ids, requests_ids, requests_ids), 0, 100, plp.LpContinuous)


    # Number of unserved demands and fast charged used
    self.n_u = plp.LpVariable("unserved", lowBound = 0, cat = plp.LpInteger)
    self.n_f = plp.LpVariable("fastcharge", lowBound = 0, cat = plp.LpInteger)

    # Instantiate the model
    self.model = plp.LpProblem(name="optim", sense=plp.LpMinimize)

    # Objective function
    self.model += plp.lpSum([self.y[v][r1][r2] * self.cost[r1, r2] + self.b[v][r1][r2] * self.pe
                            for v in range(n_aircraft) for r1 in range(n_requests) for r2 in range(n_requests)]) + self.n_u * self.lbda_u + self.n_f * self.lbda_f, "Total Costs"


    #-- Adding constraints --

    #nb of fast charge and nb of unserved demands
    self.model += self.n_u == plp.lpSum([1 -
                                        plp.lpSum([self.y[v][r1][r2] for v in range(n_aircraft) for r1 in range(n_requests) if r1 != r2])
                                        for r2 in range(n_requests) if self.requests[r2, ORIGIN] != self.requests[r2, DEST]])

    self.model += self.n_f == plp.lpSum([(1 - self.sa[v][r1][r2]) + (1 - self.sb[v][r1][r2])
                                        for v in range(n_aircraft) for r1 in range(n_requests) for r2 in range(n_requests)])


    for r1 in range(n_requests):
      for r2 in range(n_requests):
        self.model += plp.lpSum([self.y[v][r1][r2] for v in range(n_aircraft)]) <= 1
        #-- going into the past constraint
        self.model += plp.lpSum([self.y[v][r1][r2] for v in range(n_aircraft)]) <= max(0, self.requests[r2, ORIGIN_TIME] - self.requests[r1, DEST_TIME] - self.time[self.requests[r1, DEST], self.requests[r2, ORIGIN]] - self.delta)

    for r2 in range(n_requests):
      self.model += plp.lpSum([self.y[v][r1][r2] for v in range(n_aircraft) for r1 in range(n_requests) if r1 != r2]) <= 1
      for v in range(n_aircraft):
        self.model += plp.lpSum([self.y[v][r2][r1] for r1 in range(n_requests) if r1 != r2]) <= 1

    for v in range(n_aircraft):
      # startv = self.aircraft[v, 2] # starting point for evtol v
      self.model += self.e_after[v][n_requests - n_aircraft + v] == 100 #init charge
      for r1 in range(n_requests):
        self.model += self.y[v][r1][r1] == 0
        for r2 in range(n_requests):
          drr = self.delta if self.requests[r1, DEST] != self.requests[r2, ORIGIN] else 0
          #time CHARGING
          self.model += self.ta[v][r1][r2] <= max(0, self.requests[r2, ORIGIN_TIME] - self.requests[r1, DEST_TIME] - self.time[self.requests[r1, DEST], self.requests[r2, ORIGIN]] - drr)
          self.model += self.tb[v][r1][r2] <= max(0, self.requests[r2, ORIGIN_TIME] - self.requests[r1, DEST_TIME] - self.time[self.requests[r1, DEST], self.requests[r2, ORIGIN]] - drr)
          self.model += self.ta[v][r1][r2] + self.tb[v][r1][r2] <= max(0, self.requests[r2, ORIGIN_TIME] - self.requests[r1, DEST_TIME] - self.time[self.requests[r1, DEST], self.requests[r2, ORIGIN]])

          # Slow - after r1 : pas
          self.model += self.pas[v][r1][r2] <= self.ta[v][r1][r2]
          self.model += self.pas[v][r1][r2] <= self.sa[v][r1][r2] * self.big_M
          self.model += self.pas[v][r1][r2] >= self.ta[v][r1][r2] - (1 - self.sa[v][r1][r2]) * self.big_M
          # Fast - after r1 : paf
          self.model += self.paf[v][r1][r2] <= self.ta[v][r1][r2]
          self.model += self.paf[v][r1][r2] <= (1 - self.sa[v][r1][r2]) * self.big_M
          self.model += self.paf[v][r1][r2] >= self.ta[v][r1][r2] - self.sa[v][r1][r2] * self.big_M
          # Slow - before r2 : pbs
          self.model += self.pbs[v][r1][r2] <= self.tb[v][r1][r2]
          self.model += self.pbs[v][r1][r2] <= self.sb[v][r1][r2] * self.big_M
          self.model += self.pbs[v][r1][r2] >= self.tb[v][r1][r2] - (1 - self.sb[v][r1][r2]) * self.big_M
          # Fast - before r2 : pbf
          self.model += self.pbf[v][r1][r2] <= self.tb[v][r1][r2]
          self.model += self.pbf[v][r1][r2] <= (1 - self.sb[v][r1][r2]) * self.big_M
          self.model += self.pbf[v][r1][r2] >= self.tb[v][r1][r2] - self.sb[v][r1][r2] * self.big_M

          # Binary variable to indicate min (26)-(29)
          self.model += 100 - self.e_after[v][r1] - (self.gamma_s * self.pas[v][r1][r2] + self.gamma_f * self.paf[v][r1][r2])  <= self.big_M * self.za[v][r1][r2]
          self.model += (self.gamma_s * self.pas[v][r1][r2] + self.gamma_f * self.paf[v][r1][r2]) - 100 + self.e_after[v][r1] <= self.big_M * (1 - self.za[v][r1][r2])

          self.model += 100 - (self.e_after[v][r1] - self.energy[self.requests[r1, DEST], self.requests[r2, ORIGIN]] + self.ba[v][r1][r2]) - (self.gamma_s * self.pbs[v][r1][r2] + self.gamma_f * self.pbf[v][r1][r2])  <= self.big_M * self.zb[v][r1][r2]
          self.model += (self.gamma_s * self.pbs[v][r1][r2] + self.gamma_f * self.pbf[v][r1][r2]) - 100 + (self.e_after[v][r1] - self.energy[self.requests[r1, DEST], self.requests[r2, ORIGIN]] + self.ba[v][r1][r2]) <= self.big_M * (1 - self.zb[v][r1][r2])

          # Computing ba : energy purchased after r1 (30)-(34)
          self.model += self.ba[v][r1][r2] <= self.gamma_s * self.pas[v][r1][r2] + self.gamma_f * self.paf[v][r1][r2]
          self.model += self.ba[v][r1][r2] <= 100 - self.e_after[v][r1]
          self.model += self.ba[v][r1][r2] >= self.gamma_s * self.pas[v][r1][r2] + self.gamma_f * self.paf[v][r1][r2] - (1 - self.za[v][r1][r2]) * self.big_M
          self.model += self.ba[v][r1][r2] >= 100 - self.e_after[v][r1] - self.za[v][r1][r2] * self.big_M
          self.model += self.ba[v][r1][r2] <= self.y[v][r1][r2] * self.big_M

          # Computing bb : energy purchased before r2 (35)-(39)
          self.model += self.bb[v][r1][r2] <= self.gamma_s * self.pbs[v][r1][r2] + self.gamma_f * self.pbf[v][r1][r2]
          self.model += self.bb[v][r1][r2] <= 100 - (self.e_after[v][r1] - self.energy[self.requests[r1, DEST], self.requests[r2, ORIGIN]] + self.ba[v][r1][r2])
          self.model += self.bb[v][r1][r2] >= self.gamma_s * self.pbs[v][r1][r2] + self.gamma_f * self.pbf[v][r1][r2] - (1 - self.zb[v][r1][r2]) * self.big_M
          self.model += self.bb[v][r1][r2] >= 100 - (self.e_after[v][r1] - self.energy[self.requests[r1, DEST], self.requests[r2, ORIGIN]] + self.ba[v][r1][r2]) - self.zb[v][r1][r2] * self.big_M
          self.model += self.bb[v][r1][r2] <= self.y[v][r1][r2] * self.big_M

          # Computing b = ba + bb
          self.model += self.b[v][r1][r2] == self.ba[v][r1][r2] + self.bb[v][r1][r2]

          if r1 != v + n_requests - n_aircraft:
            self.model += self.y[v][r1][r2] <= plp.lpSum([self.y[v][p][r1] for p in range(n_requests) if p != r1])

          self.model += self.e_after[v][r2] >= self.e_before[v][r2] - self.energy[self.requests[r2, ORIGIN], self.requests[r2, DEST]] - self.big_M * (1 - self.y[v][r1][r2])
          self.model += self.e_after[v][r2] <= self.e_before[v][r2] - self.energy[self.requests[r2, ORIGIN], self.requests[r2, DEST]] + self.big_M * (1 - self.y[v][r1][r2])

          self.model += self.e_before[v][r2] <= self.e_after[v][r1] + self.ba[v][r1][r2] - self.energy[self.requests[r1, DEST], self.requests[r2, ORIGIN]] + self.bb[v][r1][r2] + self.big_M * (1 - self.y[v][r1][r2])
          self.model += self.e_before[v][r2] >= self.e_after[v][r1] + self.ba[v][r1][r2] - self.energy[self.requests[r1, DEST], self.requests[r2, ORIGIN]] + self.bb[v][r1][r2] - self.big_M * (1 - self.y[v][r1][r2])

          self.model += self.y[v][r1][r2] * self.min_soc <= self.e_before[v][r2]
          if self.requests[r1, DEST] != self.requests[r2, ORIGIN]:
            self.model += self.y[v][r1][r2] * self.min_soc <= self.e_after[v][r1] + self.ba[v][r1][r2]


  def solve(self, max_time, opt_gap, verbose=1):
    """ Solve the Mixed Integers Linear Program instance built in self.build_model().

      Args :
              max_time : int, maximum running time required in seconds.
              opt_gap : float, in (0, 1), if max_time is None, then the objective value
              of the solution is guaranteed to be at most opt_gap % larger than the true
              optimum.
              verbose : 1 to print log of resolution. 0 for nothing.


      Returns :
              Status of the model : Infeasible or Optimal.
              Infeasible indicates that all constraints could not be met.
              Optimal indicates that the model has been solved optimally.

    """
    start = time.time()
    self.model.solve(plp.PULP_CBC_CMD(maxSeconds = max_time, fracGap = opt_gap, msg = verbose, options=["randomCbcSeed 31"]))
    #Get Status
    logging.info(f"Status {plp.LpStatus[self.model.status]}")
    logging.info(f"Total Costs = {plp.value(self.model.objective)}")
    logging.info(f"Solving wall time : {round(time.time() - start, 3)} seconds.")
    return plp.LpStatus[self.model.status], plp.value(self.model.objective), round(time.time() - start, 3)


spec = [
    ('assignement', float64[:]),
    ('assignementMap', float64[:, :, :]),
    ('charging_times', float64[:, :, :]),
    ('energy_levels', float64[:, :, :]),
    ('energy_bought', float64[:, :, :]),
    ('violation', float64[:, :, :]),
    ('violation_tot', float64),
    ('unserved_count', int64),
    ('fast_charge_count', int64),
    ('cost', float64),
    ('routing_cost', float64),
    ('electricity_cost', float64)
]

@jitclass(spec)
class SolutionSls():
  """
      This dataclass is an object representing a solution for the routing and
      recharging problem.
      It served for the local search.

      Args:
            assignement: shape (n_requests,) of integers variables.
                         element (i, 1) is the id of the aicraft assigned to
                         request i, -1 if no aircraft is assigned to it.
            assignementMap: shape (n_request, 2, n_aircraft) of integers variables.
                            This a map that maps requests to their predecessor and successor.
                            element (i, 0, v) is the predecessor of request i in route of v, -1 if none.
                            element (i, 1, v) is the successor of request i in route of v, -1 if none.
                            The starting point of every aircraft is known using the fake requests.
            charging_times: shape (n_requests, n_aircraft, 2) of float variables.
                          element (i, j, 0) gives the charging time of aircraft j
                          after request i.
                          element (i, j, 1) gives the charging time of aircraft j
                          before request i.
                          --
                          Negative values indicate slow charge, positive fast charge.
            energy_levels: shape (n_requests, n_aircraft, 2) of float variables.
                          element (i, j, 0) gives the energy level of aircraft j
                          after request i.
                          element (i, j, 1) gives the energy level of aircraft j
                          before request i.
            energy_bought: shape (n_requests, n_aircraft, 2) of float variables.
                          element (i, j, 0) gives the energy bought by aircraft j
                          after request i.
                          element (i, j, 1) gives the energy bought by aircraft j
                          before request i.
            violation: shape (n_requests, n_aircraft, 2) of float variables.
                          element (i, j, 0) gives the battery soc violation of aircraft j
                          after request i.
                          element (i, j, 1) gives the battery soc violation of aircraft j
                          before request i.
            violation_tot: overall battery violation value
            unserved_count: number of unserved demands in solution.
            cost: total cost of the solution
            routing_cost: total routing cost
            electricity_cost: total cost for buying electricity
  """
  def __init__(self,
               assignement: np.ndarray,
               assignementMap: np.ndarray,
               charging_times: np.ndarray,
               energy_levels: np.ndarray,
               energy_bought: np.ndarray,
               violation: np.ndarray,
               violation_tot: float,
               unserved_count: int,
               fast_charge_count: int,
               cost: float,
               routing_cost: float,
               electricity_cost: float):
    self.assignement = assignement
    self.assignementMap = assignementMap
    self.charging_times = charging_times
    self.energy_levels = energy_levels
    self.energy_bought = energy_bought
    self.violation = violation
    self.violation_tot = violation_tot
    self.unserved_count = unserved_count
    self.fast_charge_count = fast_charge_count
    self.cost = cost
    self.routing_cost = routing_cost
    self.electricity_cost = electricity_cost

  def get_energy_bought_after(self, r: int, v: int):
    """ Gives the energy bought by aircraft v after request r.

        Args:
              r: request id
              v: aircraft id

        Returns:
              float, violation value
    """
    return self.energy_bought[r, v, 0]

  def get_energy_bought_before(self, r: int, v: int):
    """ Gives the energy bought by aircraft v before request r.

        Args:
              r: request id
              v: aircraft id

        Returns:
              float, violation value
    """
    return self.energy_bought[r, v, 1]

  def get_violation_after(self, r: int, v: int):
    """ Gives the battery soc violation of aircraft v after request r.

        Args:
              r: request id
              v: aircraft id

        Returns:
              float, violation value
    """
    return self.violation[r, v, 0]

  def get_violation_before(self, r: int, v: int):
    """ Gives the battery soc violation of aircraft v before request r.

        Args:
              r: request id
              v: aircraft id

        Returns:
              float, violation value
    """
    return self.violation[r, v, 1]

  def get_charging_time_after(self, r: int, v: int):
    """ Gives the charging time of aircraft v after request r.

        Args:
              r: request id
              v: aircraft id

        Returns:
              float, violation value
    """
    return self.charging_times[r, v, 0]

  def get_charging_time_before(self, r: int, v: int):
    """ Gives the charging time of aircraft v before request r.

        Args:
              r: request id
              v: aircraft id

        Returns:
              float, violation value
    """
    return self.charging_times[r, v, 1]

  def get_succ(self, r: int, v: int):
    """ Gives the successor of request r in the route
        of v.

        Args:
            r: request id
            v: aircraft id

        Returns:
            int, id of request successor

        Raises:
            ValueError if r is not served by v.
    """
    if self.assignement[r] != v:
      raise ValueError("Request  is not served by aircraft .")
    return int(self.assignementMap[r, 1, v])

  def get_pred(self, r: int, v: int):
    """ Gives the predecessor of request r in the route
        of v.

        Args:
            r: request id
            v: aircraft id

        Returns:
            int, id of request predecessor

        Raises:
            ValueError if r is not served by v.
    """
    if self.assignement[r] != v:
      print(r, v)
      raise ValueError("Request  is not served by aircraft .")
    return int(self.assignementMap[r, 0, v])

  def get_assigned_aircraft(self, r: int):
    """ Gives the aircraft assigned to request r.

        Args:
            r: request id

        Returns:
            int, aircraft id serving r
    """
    return int(self.assignement[r])

  def get_energy_after(self, r: int, v: int):
    """ Gives the energy of aicraft v after request r.

        Args:
            r: request id
            v: aircraft id

        Returns:
            float, aircraft id serving r
    """
    return self.energy_levels[r, v, 0]

  def get_energy_before(self, r: int, v: int):
    """ Gives the energy of aicraft v before request r.

        Args:
            r: request id
            v: aircraft id

        Returns:
            float, aircraft id serving r
    """
    return self.energy_levels[r, v, 1]

  def commit_new_assignment(self, r: int, v: int):
    """ Assigns request r to aircraft v

        Args:
            r: request id
            v: aircraft id

        Returns:
            None
    """
    self.assignement[r] = v

  def commit_new_succ(self, r: int, v: int, succ: int):
    """ Puts succ as successor of r in route of v

        Args:
            r: request id
            v: aircraft id
            succ: request id of successor

        Returns:
            None
    """
    self.assignementMap[r, 1, v] = succ

  def commit_new_pred(self, r: int, v: int, pred: int):
    """ Puts pred as predecessor of r in route of v

        Args:
            r: request id
            v: aircraft id
            pred: request id of predecessor

        Returns:
            None
    """
    self.assignementMap[r, 0, v] = pred

  def commit_new_charge_time_after(self, r: int, v: int, ta: float):
    """ Puts ta as new charge time after r in route of v

        Args:
            r: request id
            v: aircraft id
            ta: time to charge battery

        Returns:
            None
    """
    self.charging_times[r, v, 0] = ta

  def commit_new_charge_time_before(self, r: int, v: int, tb: float):
    """ Puts tb as new charge time before r in route of v

        Args:
            r: request id
            v: aircraft id
            tb: time to charge battery

        Returns:
            None
    """
    self.charging_times[r, v, 1] = tb

  def commit_new_energy_before(self, r: int, v: int, eb: float):
    """ Puts eb as energy of aircraft v before request r.

        Args:
            r: request id
            v: aircraft id
            eb: new energy level

        Returns:
            float, aircraft id serving r
    """
    self.energy_levels[r, v, 1] = eb

  def commit_new_energy_after(self, r: int, v: int, ea: float):
    """ Puts ea as energy of aircraft v after request r.

        Args:
            r: request id
            v: aircraft id
            ea: new energy level

        Returns:
            float, aircraft id serving r
    """
    self.energy_levels[r, v, 0] = ea

  def commit_new_energy_bought_after(self, r: int, v: int, eb: float):
    """ Puts eb as energy bought by aircraft v after request r.

        Args:
            r: request id
            v: aircraft id
            eb: new energy bought

        Returns:
            float, aircraft id serving r
    """
    self.energy_bought[r, v, 0] = eb

  def commit_new_energy_bought_before(self, r: int, v: int, eb: float):
    """ Puts eb as energy bought by aircraft v before request r.

        Args:
            r: request id
            v: aircraft id
            eb: new energy bought

        Returns:
            float, aircraft id serving r
    """
    self.energy_bought[r, v, 1] = eb

  def commit_new_violation_after(self, r: int, v: int, va: float):
    """ Puts va as the battery soc violation of aircraft v after request r.

        Args:
              r: request id
              v: aircraft id

        Returns:
              None
    """
    self.violation[r, v, 0] = va

  def commit_new_violation_before(self, r: int, v: int, vb: float):
    """ Puts vb the as battery soc violation of aircraft v before request r.

        Args:
              r: request id
              v: aircraft id
              vb: new violation value

        Returns:
              None
    """
    self.violation[r, v, 1] = vb



@njit(inline="always")
def VND(current_sol,
        staging_sol,
        random_sol,
        move,
        problem_instance,
        M,
        counts,
        penalty=0.,
        first_best=False,
        apply_neigh=apply_neigh):
  """ Variable Neighbourhood Descent for the VNS algo.

  """
  neigh_idx = 0
  tour = 0
  while neigh_idx < M:
    tour += 1
    if tour > 2000:
      raise RuntimeError("Suspect behavior in VND. Likely stuck in oscillation.")
    counts[neigh_idx] = counts[neigh_idx] + 1
    #-- shuffling request and aircraft ids
    np.random.shuffle(problem_instance.requests_ids)
    np.random.shuffle(problem_instance.aircraft_ids)
    #-- apply current neighbouhood
    reward, move = apply_neigh(neigh_idx,
                              random_sol,
                              current_sol,
                              staging_sol,
                              problem_instance,
                              move,
                              penalty=penalty,
                              first_best=first_best)

    # -- Going to next neighbourhood
    if reward > 0:
      neigh_idx = 0
    else:
      neigh_idx += 1

  return move, counts



@njit(parallel=False)
def GVNS(current_sol: object,
        best_sol: object,
        staging_sol: object,
        random_sol,
        n_iter: int,
        timeout: float,
        problem_instance: object,
        seed: int = 7,
        verbose: bool = True,
        no_imp_limit: int = 250,
        J: int = 50,
        incr: float = 5,
        decr: float = 2,
        first_best: bool = False,
        W: int = 10,
        copy_solution=copy_solution,
        update_penalty=update_penalty,
        shake=shake,
        VND=VND,
        SAT_stop:bool = False):
  """ Implement a variable neighbourhoods search (VNS),
      the search alternates between using different neighbourhoods operators
      switching whenever no progress is made. A MAB is used to determine which
      neighborhood is chosen each time stagnation happens.

      Args :
              current_sol: solution instance of current
              best_sol: solution instance containing best obtained
              staging_sol: solution instance, used for staging a temporary sol
              n_iter: maximum number of iterations
              verbose: whether to print progress info to stdout
              mab_eps: probability for exploration in the MAB
              no_imp_limit: number of stagnation iteration after which the algo should stop
              incr: increment for the penalty on energy (additive)
              decr: decrement for the penalty on energy (multiplicative)

      Returns :
              best_sol: np.ndarray, encoding of valid solution - best found during the search
              best_cost: float, cost of best_sol
              cache_cost_best: list, best_cost encountered during the search, in order
              cache_cost_current: list, cost of solution explored during the search, in order
              perf_over_time: list[Tuple], contains sequence of best cost obtained with time it took : useful for anytime aspect
  """
  np.random.seed(seed)
  copy_solution(current_sol, best_sol)
  copy_solution(current_sol, staging_sol)
  copy_solution(current_sol, random_sol)
  shake_neigh = [0, 3]
  neigh_idx = 0
  fail_curr = 0
  counts = np.array([0, 0, 0, 0])
  rolling_violations = np.zeros((W,))
  anytime_cache = np.zeros((1, 4))
  stag = 0
  pen_energy = 0
  M = 4
  move = 0
  # print("Starting loop...")
  for it in range(n_iter):
    if it == 1:
      #---
      # Initiate time at iteration 1 to avoid compilation time.
      # This results in an approximative measure as the first iteration
      # will not count in the total time. However, this time should not
      # be significant (~ 1-3 seconds at most)
      #---
      with objmode(time_start='f8'):
        time_start = time.time()

    stag += 1
    if stag > no_imp_limit and best_sol.violation_tot < 0.9:
      break
    if (1 + it) % J == 0:
      # Every J iterations, a new penalty is computed
      new_pen_energy = update_penalty(pen_energy,
                                      rolling_violations,
                                      incr,
                                      decr)
      # Increase penalty if there is a stagnation in infeasbility
      if best_sol.violation_tot > 0.9 and new_pen_energy >= 5000:
        print("Boosting violation penalty")
        #no best has been found yet
        #boost increase to ensure a valid solution will be found
        new_pen_energy *= 2
      #-- Update cost attributes.
      current_sol.cost += current_sol.violation_tot * (new_pen_energy - pen_energy)
      best_sol.cost += best_sol.violation_tot * (new_pen_energy - pen_energy)
      # Update penalty
      pen_energy = new_pen_energy
    if it % 50 == 0 and verbose:
      print("Iteration : ", it)
      print("Current cost is ", current_sol.cost)
      print("Best cost is ", best_sol.cost)
      print("Penalty energy: ", pen_energy)
      print("Counts: ", counts)

    #-- Shake current solution
    copy_solution(current_sol, random_sol)
    move = shake(neigh_idx,
                 random_sol,
                 staging_sol,
                 problem_instance,
                 move,
                 penalty=pen_energy)
    counts[neigh_idx] += 1

    #-- Apply VND around the obtained point
    move, counts = VND(current_sol,
                        staging_sol,
                        random_sol,
                        move,
                        problem_instance,
                        M,
                        counts,
                        penalty=pen_energy,
                        first_best=first_best)

    neigh_idx = 2 if neigh_idx == 0 else 0

    with objmode(current_time='f8'):
      current_time = time.time() - time_start

    if current_sol.cost < best_sol.cost and current_sol.violation_tot < 0.2:
      copy_solution(current_sol, best_sol)
      #reset shake neighbourhood
      neigh_idx = 0
      #reset stagnation count
      stag = 0
      #-- Caching best_sol for anytime monitoring.
      aux = np.zeros((1, 4))
      aux[0, 0] = 1 - (best_sol.unserved_count / problem_instance.n_requests)
      aux[0, 1] = float(best_sol.fast_charge_count)
      aux[0, 2] = (best_sol.routing_cost + best_sol.electricity_cost) / (problem_instance.n_requests - best_sol.unserved_count)
      aux[0, 3] = current_time
      anytime_cache = np.concatenate((anytime_cache, aux), axis=0)
      if SAT_stop and best_sol.unserved_count == 0:
        # print("Breaking")
        break

    #--
    rolling_violations[it % W] = current_sol.violation_tot

    if it > 2 and current_time > timeout:
      #timeout break
      break

  with objmode(search_time='f8'):
    search_time = round(time.time() - time_start, 2)

  return pen_energy, search_time, move, anytime_cache





def run_gurobi(model_path: str,
               max_time: float,
               output_file: str,
               relGap: float = 0.001):
  """ Calls Gurobi through minizinc to solve model.

      Args:
          model_path: path to mzn file containing
          both model and data.
          ex: src/logs_results/instances_data/Model_BENCHMARK_3_3_30_7.mzn
          max_time: maximum solving time for gurobi,
          in seconds.

      Returns:
          flat_time: time it took for minizinc to compile


  """
  command = f"minizinc -i -o '{output_file}' -p 4 -r 7 --solver-time-limit {max_time * 1000} --relGap {relGap} --output-time --compiler-statistics --verbose-solving --no-optimize --solver gurobi {model_path} 2> {output_file + '_logs'}"
  stream = os.popen(command)
  output = stream.read()
  rows = output.splitlines()
  flat_time = "NC"
  for row in rows:
    if "flatTime" in row:
      flat_time = float(row.split("=")[-1])
  print(f"Flattening time is {flat_time}")
  print(f"Solution stream outputed to {output_file}")
  with open(output_file, "a") as file_object:
    file_object.write(f"\n Flat time {flat_time}.")
  return flat_time



