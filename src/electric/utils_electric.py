import numpy as np
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass, field
import heapq
from numba import njit, jit, int64, float64
from numba.experimental import jitclass
import numba
import sys
import re

# from electric.routing_charging import neighborhoods
# from electric.routing_charging.optim_electric import SolutionSls

ORIGIN = 1
DEST = 2
ORIGIN_TIME = 3
DEST_TIME = 4

def simulate_infra(s: int, v: int, r: int, T: list, delta):
  """
      Simulate problem instances to test electric milp

      Args:
          s : int, number of skyports
          v : int, number of evtols
          r : int, number of requests
          seed: int, seed for random infra
          T: list, sequence of time steps.

      Returns:
          cost: np.ndarray, table containing $ cost to link request i to j
          time: np.ndarray, table containing time to travel from i to j
          energy: np.ndarray, table containing energy used %SoC to travel from i to j
          requests: np.ndarray, table containing request info : id | r- | r+ | t_r- | t_r+ | beta_r
          aircraft: np.ndarray, table containing aircraft carac : id | eta | start_id |
          skyports: np.ndarray, table containing skyport carac : id | landing fee |
  """
  ORIGIN = 1
  DEST = 2
  ORIGIN_TIME = 3
  DEST_TIME = 4
  np.random.seed(7)
  eta = 34
  skyports = np.zeros((s, 2), dtype=np.int64)
  skyports[:, 0] = range(s)
  skyports[:, 1] = np.random.choice([30, 40, 80], s)
  aircraft = np.zeros((v, 3), dtype=np.int64)
  aircraft[:, 0] = range(v)
  aircraft[:, 1] = eta
  aircraft[:, 2] = np.random.choice(range(s), v)
  time = np.random.choice([0, 10, 15, 15, 20, 20], (s, s))
  time = np.maximum(time, time.transpose())  #ensuring symetry
  time[time == 0] = 10
  np.fill_diagonal(time, 0)
  energy = time.copy().astype('float64')
  energy[time == 10] = 15
  energy[time == 15] = 20
  energy[time == 20] = 25
  np.fill_diagonal(energy, 0)
  requests = np.zeros((r + v, 6), dtype=np.int64)
  requests[:, 0] = range(r + v)
  cs = int((len(T)) / r)
  print(cs)
  for i in range(r):
    ori, dest = np.random.choice(range(s), size=2, replace=False)
    #beginning of service
    if r > 120:
      j = int(cs * i)
      if j > 690:
        j = 690
      e_time = np.random.choice(T[j:j + cs])
    else:
      e_time = np.random.choice([k for k in range(int(len(T[10:-30]) / (r)) * (i), int(len(T[10:-30]) / r) * (i + 1))])
    print(f"Demand {i} from {ori} to {dest} at {'{:02d}:{:02d}'.format(*divmod(e_time + 420, 60))}")
    #end of service
    l_time = e_time + time[ori, dest]
    requests[i, ORIGIN] = ori
    requests[i, DEST] = dest
    requests[i, ORIGIN_TIME] = e_time
    requests[i, DEST_TIME] = l_time
  #gain for serving request is randomly gen but may depend on pooling phase
  # 80$ per passenger either 2, 3 or 4 pax
  requests[:r, 5] = np.random.choice([160, 240, 320], r)
  #adding starting fake request one per vtol
  for i in range(v):
    requests[r+i, 1] = aircraft[i, 2]
    requests[r+i, 2] = aircraft[i, 2]
    requests[r+i, 3] = 0
    requests[r+i, 4] = 0
  # Now computing cost to connect demands...
  # ---
  costs = np.zeros((r+v, r+v))
  for i in range(r+v):
    for j in range(r+v):
      if i != j:
        drr = delta if requests[i, DEST] == requests[j, ORIGIN] else 2 * delta
        feas = requests[j, 3] - requests[i, 4] + time[int(requests[i, 2]), int(requests[j, 1])] - drr > 0
        if feas:
          if requests[i, 2] != requests[j, 1]:
            costs[i, j] = eta * (time[int(requests[i, 2]), int(requests[j, 1])] + time[int(requests[j, 1]), int(requests[j, 2])]) + skyports[int(requests[j, 1]), 1] + skyports[int(requests[j, 2]), 1] - requests[j, 5]
          else:
            costs[i, j] = eta * time[int(requests[j, 1]), int(requests[j, 2])] + skyports[int(requests[j, 2]), 1] - requests[j, 5]
        else:
          costs[i, j] = 0
  #--- return problem vars
  return costs, time, energy, requests, aircraft, skyports

def format_time(t):
  """ Format time in HH:MM format - human readable time.
      Args:
          t: int, between 0 and 720.

      Returns:
          str, HH:MM time format of t
  """
  return f"{'{:02d}:{:02d}'.format(*divmod(t + 420, 60))}"

def read_milp_sol(model, obj):
  """
      Reads a milp solution by reading decisions variable values.
      Saves a plot of charging patterns of fleet.
      This function will be used primarly to test the model (not at scale).

      Args :
            model: ElectricMilp instance, solved
            obj: float, objective value of the solved optim.

      Returns:
            None

  """
  ORIGIN = 1
  DEST = 2
  ORIGIN_TIME = 3
  DEST_TIME = 4
  e_bought = 0
  e_trace = {}
  for v in range(model.aircraft.shape[0]):
    print("")
    print(f"--- Aircraft {v} ---")
    tt = []
    td = {}
    e_trace[v] = {}
    e_trace[v]['energy'] = []
    e_trace[v]["time"] = []
    for r1 in range(model.requests.shape[0]):
      for r2 in range(model.requests.shape[0]):
        if model.y[v][r1][r2].varValue == 1:
          tt.append(model.requests[r1, DEST_TIME])
          txt = "\n"
          txt += f"Aircraft {v} is serving {r1} and {r2} \n"
          txt += f"Aircraft {v} is at {model.requests[r1, DEST]} at {format_time(model.requests[r1, DEST_TIME])} - {model.e_after[v][r1].varValue} % energy. \n"
          txt += f"charging for {model.ta[v][r1][r2].varValue} minutes in mode : {(1 - int(model.sa[v][r1][r2].varValue)) * 'fast' + int(model.sa[v][r1][r2].varValue) * 'slow'} - Leaving to {model.requests[r1, 1]}, arriving at {format_time(int(model.requests[r1, DEST] + model.ta[v][r1][r2].varValue + model.time[model.requests[r1, DEST], model.requests[r2, ORIGIN]]))}. \n"
          txt += f"charging for {model.tb[v][r1][r2].varValue} minutes in mode : {(1 - int(model.sb[v][r1][r2].varValue)) * 'fast' + int(model.sb[v][r1][r2].varValue) * 'slow'} \n"
          txt += f"Serving {r2} - arriving at {model.requests[r2, DEST]} at {format_time(model.requests[r2, DEST_TIME])} - {model.e_after[v][r2].varValue} % energy. "
          td[model.requests[r1, DEST_TIME]] = txt

          e_trace[v]['energy'] += [model.e_after[v][r1].varValue,
                                   model.ba[v][r1][r2].varValue + model.e_after[v][r1].varValue,
                                   model.e_before[v][r2].varValue - model.bb[v][r1][r2].varValue,
                                   model.e_before[v][r2].varValue,
                                   model.e_before[v][r2].varValue,
                                   model.e_after[v][r2].varValue]

          e_trace[v]['time'] += [model.requests[r1, DEST_TIME],
                                 model.requests[r1, DEST_TIME] + model.ta[v][r1][r2].varValue,
                                 model.requests[r1, DEST_TIME] + model.ta[v][r1][r2].varValue + model.time[model.requests[r1, DEST], model.requests[r2, ORIGIN]],
                                 model.requests[r1, DEST_TIME] + model.ta[v][r1][r2].varValue + model.time[model.requests[r1, DEST], model.requests[r2, ORIGIN]] + model.tb[v][r1][r2].varValue  + model.tb[v][r1][r2].varValue,
                                 model.requests[r2, ORIGIN_TIME],
                                 model.requests[r2, DEST_TIME]]

          e_bought += model.b[v][r1][r2].varValue

    tt.sort()
    for j in tt:
      print(td[j])
    print("")
  print(f"Cost of solution : ${obj} || Operating cost : ${obj - model.pe * e_bought - model.n_u.varValue * model.lbda_u - model.n_f.varValue * model.lbda_f}, Electricity cost ${model.pe * e_bought}, Penalty demands {model.n_u.varValue * model.lbda_u}, Penalty fast charge {model.n_f.varValue * model.lbda_f}")
  plt.figure(figsize=(18, 10))
  for v in range(model.aircraft.shape[0]):
    plt.plot(sorted(e_trace[v]['time']), [x for _,x in sorted(zip(e_trace[v]['time'], e_trace[v]['energy']))], label=f"eVTOL {v}")
  plt.axhline(y=30, color='r', linestyle='--')
  plt.xlabel("Time")
  plt.xticks([i for i in range(len(model.T))][::30], ['{:02d}:{:02d}'.format(*divmod(420 + i, 60)) for i in range(len(model.T))][::30])
  plt.yticks([0, 100], ["SoC Low", "SoC High"])
  plt.ylabel("SoC in %")
  plt.title("State of Charge of Fleet of eVTOLs over one day of service.")
  plt.legend()
  plt.savefig('/results/Electric_plots/Charge_SoC_eVTOLs.png')

@njit(cache=True)
def process_compatibility(requests: np.ndarray, aircraft: np.ndarray, time: np.ndarray, delta: float):
  """
      Compute a 2D table indicating whether two requests can be chained or
      if an aircraft can start day with a request.

      Args :
            requests: np.ndarray, table containing request info : id | r- | r+ | t_r- | t_r+ | beta_r
            aircraft: np.ndarray, table containing aircraft carac : id | eta | start_id |

      Returns:
            time_compatible: np.ndarray,

  """
  ORIGIN = 1
  DEST = 2
  ORIGIN_TIME = 3
  DEST_TIME = 4
  size = requests.shape[0]
  time_compatible = np.zeros((size, size))
  for i in range(size):
    for j in range(size):
      drr = delta if requests[i, DEST] == requests[j, ORIGIN] else 2 * delta
      if requests[j, ORIGIN_TIME] - requests[i, DEST_TIME] - time[requests[i, DEST], requests[j, ORIGIN]] - drr > 0:
        time_compatible[i, j] = 1
  return time_compatible

def get_problem_instance(s=3, v=3, r=10, min_soc=55., soc_max=92., delta=10., pe=1.):
  """ Filling problem instance object.

      This is done this way to be able to use jitclass on problem instance.

  """
  pi=ProblemInstance()
  pi.seed = 7
  pi.s=s
  pi.v=v
  pi.r=r
  pi.min_soc=min_soc
  pi.soc_max=soc_max
  pi.delta=delta
  pi.pe=pe
  pi.n_requests=r
  pi.n_aircraft=v
  pi.aircraft_ids=np.arange(v)
  pi.ids_aicraft = np.arange(v)
  pi.requests_ids = np.arange(r)
  (pi.costs,
    # pi.landing_costs,
    pi.time,
    pi.energy,
    pi.requests,
    pi.aircraft,
    pi.skyports) = simulate_infra(pi.s,
                                      pi.v,
                                      pi.r,
                                      pi.T,
                                      pi.delta)
  pi.time_compatible = process_compatibility(pi.requests,
                                                  pi.aircraft,
                                                  pi.time,
                                                  pi.delta)

  #-- penalties for service and fast charges
  pi.lbda_f = compute_fastcharge_coef(pi)
  pi.lbda_u = compute_service_coef(pi)
  return pi

spec = [
    ('delta', float64),
    ('gamma_s', float64),
    ('gamma_f', float64),
    ('pe', float64),
    ('lbda_u', float64),
    ('lbda_f', float64),
    ('min_soc', float64),
    ('s', int64),
    ('v', int64),
    ('r', int64),
    ('seed', int64),
    ('soc_max', float64),
    ('ORIGIN', int64),
    ('DEST', int64),
    ('T', numba.typeof(numba.typed.List([2]))),
    ('n_requests', int64),
    ('n_aircraft', int64),
    ('costs', float64[:,:]),
    ('ids_aicraft', int64[:]),
    ('time', int64[:, :]),
    ('energy', float64[:, :]),
    ('requests', int64[:, :]),
    ('aircraft', int64[:, :]),
    ('skyports', int64[:,:]),
    ('aircraft_ids', int64[:]),
    ('requests_ids', int64[:]),
    ('time_compatible', float64[:,:]),
    ('ORIGIN_TIME', int64),
    ('DEST_TIME', int64)
]

@jitclass(spec)
class ProblemInstance:
  """
      Container class for a routing problem instance.
      Contains properties of a problem, with setters and getters.

      Attributes:

                delta: turnaround time
                gamma_s: slow charging rate
                gamma_f: fast charging rate
                pe: $ price of kwh
                lbda_u: penalty for unserved demands
                lbda_f: penalty for using fast charge
                min_soc: min soc allowed at takeoff
                T: time steps
                costs: table of costs to link two demands
                time: table of time to fly between to skyports
                energy: table of energy required to fly between two skyports
                requests: table of requests
                aircraft: table of aircraft
                skyports: table of skyports
                time_compatble: table of demand comptatible to be served one after the other.
  """
  def __init__(self):
    """ Init class

    """
    self.delta = 10.
    self.gamma_s = 1.
    self.gamma_f = 2.
    self.pe = 1.
    self.lbda_u = 3000.
    self.lbda_f = 2.
    self.min_soc = 55.
    self.s = 3
    self.v = 3
    self.r = 10
    self.seed = 6
    self.soc_max = 92.
    self.ORIGIN = 1
    self.DEST = 2
    self.T = numba.typed.List(list(range(720)))
    self.n_requests = 3
    self.n_aircraft = 3
    self.costs = np.zeros((2, 2))
    # self.landing_costs = np.zeros((2, 2))
    self.time = np.zeros((2, 2), dtype=np.int64)
    self.energy = np.zeros((2, 2), dtype=np.float64)
    self.requests = np.zeros((2, 2), dtype=np.int64)
    self.aircraft = np.zeros((2, 2), dtype=np.int64)
    self.skyports = np.zeros((2, 2), dtype=np.int64)
    self.ids_aicraft = np.arange(3)
    self.aircraft_ids = np.arange(3)
    self.requests_ids = np.arange(10)
    self.time_compatible = np.zeros((2, 2))
    self.ORIGIN_TIME = 3
    self.DEST_TIME = 4

  def get_service_energy(self, r):
    """ Gets the energy used to serve r

        Args:
            r: id of request

        Returns:
            float, energy required to serve r
    """
    return self.energy[self.requests[r, self.ORIGIN], self.requests[r, self.DEST]]

  def get_connection_energy(self, pred, succ):
    """ Gets the energy used to connect pred and succ.

        Args:
            pred: id of request pred
            succ: if of request succ

        Returns:
            float, energy required to connect pred to succ.
    """
    return self.energy[self.requests[pred, self.DEST], self.requests[succ, self.ORIGIN]]

  def get_connection_time(self, pred, succ):
    """ Gets the time needed to connect pred and succ.

        Args:
            pred: id of request pred
            succ: if of request succ

        Returns:
            int, time required to connect pred to succ.
    """
    return self.time[self.requests[pred, self.DEST], self.requests[succ, self.ORIGIN]]

  def get_idle_time(self, pred, succ):
    """ Gets the idle time available when connecting pred and succ.

        Args:
            pred: id of request pred
            succ: if of request succ

        Returns:
            int, idle time.
    """
    return self.requests[succ, self.ORIGIN_TIME] - self.requests[pred, self.DEST_TIME] - self.get_connection_time(pred, succ)

  def get_max_charge_time(self, pred: int, succ: int):
    """ Gets the maximum possible charging time after or before
        when connecting pred and succ.

        Args:
            pred: id of request pred
            succ: if of request succ

        Returns:
            float, max charging time.
    """
    con_time = self.get_connection_time(pred, succ)
    if con_time > 0:
      return self.requests[succ, self.ORIGIN_TIME] - self.requests[pred, self.DEST_TIME] - con_time - self.delta
    return self.requests[succ, self.ORIGIN_TIME] - self.requests[pred, self.DEST_TIME]


def heap_insertion(q, k, cost, sol, i):
  """
      Insertion sort using heapq

      Args:
          q: list, queue in which cost and solution are stored
          k: int, maximum lenght allowed for q
          cost: float, cost of current sol to be inserted
          sol: np.ndarray, encoding of current sol to be inserted
          i: int, current unique index (used to ensure items can always be sorted)

      Returns:
          q: updated list.
  """
  if cost < np.inf:
    if len(q) < k:
      heapq.heappush(q, (-cost, i, sol))
    else:
      _ = heapq.heappushpop(q, (-cost, i, sol))
  return q


@njit(inline="always")
def feasible_connection(predecessor: int,
                        successor: int,
                        time_compatible: np.ndarray):
  """ Returns connection feasibility.

      If prev and succ cannot be served one after the other,
      connection is not feasible.

      Args:
            prev: id predecessor request
            succ: id of successor request
            time_compatible: compatibility table
      Returns:
            bool, true iif connection is feasible
  """
  return time_compatible[predecessor, successor] == 1

# @njit(inline="always")
def initiate_solution(s, v, r, soc_max, solution_object):
  """
      Instanciate a SolutionSls object with inital values.

      Args:
          s: number of skyports
          v: number of aircraft
          r: number of requests

      Returns:
          SolutionSls object.
  """
  assignment = np.zeros((r + v)) - 1  #all assigned to -1, i.e. not served
  assignmentMap = np.zeros((r + v, 2, v)) - 1
  charging_times = np.zeros((r + v, v, 2))
  energy_levels = np.zeros((r + v, v, 2))
  energy_bought = np.zeros((r + v, v, 2))
  violation = np.zeros((r + v, v, 2))
  # set values for fake requests.
  for i in range(v):
    assignment[r + i] = i  #vtol assigned to their respective fake requests
    energy_levels[r + i, i, 0] = soc_max
    energy_levels[r + i, i, 1] = soc_max
  violation_tot = 0.
  unserved_count = r
  cost = 0.
  routing_cost = 0.
  electricity_cost = 0.
  fast_charge_count = 0
  solution = solution_object(assignment,
                                        assignmentMap,
                                        charging_times,
                                        energy_levels,
                                        energy_bought,
                                        violation,
                                        violation_tot,
                                        unserved_count,
                                        fast_charge_count,
                                        cost,
                                        routing_cost,
                                        electricity_cost)
  return solution

def read_sls_solution(sol: object, problem_instance: ProblemInstance, name: str):
  """ Reads a SolutionSls instance and plots energy levels of fleet.

      Args:
          sol: instance of SolutionSls
          problem_instance: instance of ProblemInstance
          name: name extension for plot to be saved.

      Returns:
          None

      Saves:
          Figure of energy levels in
          '//src/logs_results/Electric_plots/Charge_SoC_eVTOLs_SLS_{name}.png'
  """
  ORIGIN = 1
  DEST = 2
  ORIGIN_TIME = 3
  DEST_TIME = 4
  energy_history = {v: {"Energy": [], "Time": []} for v in range(problem_instance.n_aircraft)}
  for v in range(problem_instance.n_aircraft):
    print(f"\n --- Route of aircraft {v} --- \n")
    predecessor = problem_instance.n_requests + v
    successor = sol.get_succ(predecessor, v)
    while predecessor >= 0:
      print(f"Connection {predecessor} -> {successor}")
      time_before_predecessor = problem_instance.requests[predecessor, ORIGIN_TIME]
      time_after_predecessor = problem_instance.requests[predecessor, DEST_TIME]
      energy_history[v]["Time"].append(time_before_predecessor)
      energy_history[v]["Energy"].append(round(sol.get_energy_before(predecessor, v), 3))
      print(f"Aircraft {v} is before {predecessor}, at {format_time(int(time_before_predecessor))} with SoC {round(sol.get_energy_before(predecessor, v), 3)} %")
      print(f"Aircraft {v} finishes serving {predecessor} at {format_time(int(time_after_predecessor))} with SoC {round(sol.get_energy_after(predecessor, v), 3)} %")
      energy_history[v]["Time"].append(time_after_predecessor)
      energy_history[v]["Energy"].append(round(sol.get_energy_after(predecessor, v), 3))
      if successor < 0:
        break
      ta = sol.get_charging_time_after(predecessor, v)
      mode = "fast" if ta > 0 else "slow"
      ta = abs(ta) #switching to positive values for plots and log
      print(f"Aircraft {v} is charging for {ta} minutes in {mode} mode. SoC gets to {round(sol.get_energy_after(predecessor, v) + sol.get_energy_bought_after(predecessor, v), 3)} % ")
      energy_history[v]["Time"].append(time_after_predecessor + ta)
      energy_history[v]["Energy"].append(round(sol.get_energy_after(predecessor, v) + sol.get_energy_bought_after(predecessor, v), 3))
      energy_dh = problem_instance.get_connection_energy(predecessor, successor)
      time_dh = problem_instance.get_connection_time(predecessor, successor)
      if ta < problem_instance.delta and energy_dh > 0:
        #deadhead happends after spending delta on ground
        energy_history[v]["Time"].append(time_after_predecessor + problem_instance.delta)
        energy_history[v]["Energy"].append(energy_history[v]["Energy"][-1])

        energy_history[v]["Time"].append(time_after_predecessor + problem_instance.delta + time_dh)
        energy_history[v]["Energy"].append(energy_history[v]["Energy"][-1] - energy_dh)
      if energy_dh > 0:
        print(f"Aircraft {v} is deadheading to demand {successor} at {format_time(int(max(ta, problem_instance.delta) + time_after_predecessor))}. Arrives at {format_time(int(max(ta, problem_instance.delta) + time_dh + time_after_predecessor))}. SoC is {round(sol.get_energy_after(predecessor, v) + sol.get_energy_bought_after(predecessor, v) - energy_dh, 3)} %.")
        energy_history[v]["Time"].append(time_after_predecessor + max(ta, problem_instance.delta) + time_dh)
        energy_history[v]["Energy"].append(round(sol.get_energy_after(predecessor, v) + sol.get_energy_bought_after(predecessor, v) - energy_dh, 3))

      tb = sol.get_charging_time_before(successor, v)
      mode = "fast" if tb > 0 else "slow"
      tb = abs(tb) # switching to positive values
      time_before_successor = problem_instance.requests[successor, ORIGIN_TIME]
      if energy_dh > 0:
        energy_history[v]["Time"].append(time_after_predecessor + max(ta, problem_instance.delta) + time_dh + tb)
        energy_history[v]["Energy"].append(round(sol.get_energy_before(successor, v), 3))
      else:
        energy_history[v]["Time"].append(time_after_predecessor + ta + time_dh + tb)
        energy_history[v]["Energy"].append(round(sol.get_energy_before(successor, v), 3))

      print(f"Aircraft {v} is charging for {tb} minutes in mode {mode}. SoC gets to {round(sol.get_energy_before(successor, v), 3)} %. \n")
      energy_history[v]["Time"].append(time_before_successor)
      energy_history[v]["Energy"].append(round(sol.get_energy_before(successor, v), 3))
      predecessor = successor
      if predecessor < 0:
        continue
      successor = sol.get_succ(predecessor, v)
  # Plot energy levels
  plt.figure(figsize=(18, 10))
  for v in energy_history:
    plt.plot(energy_history[v]['Time'], energy_history[v]['Energy'], label=f"eVTOL {v}")
  plt.axhline(y=problem_instance.min_soc, color='r', linestyle='--', label="Minimum SoC for Takeoff")
  plt.xlabel("Time")
  plt.xticks([i for i in range(len(problem_instance.T))][::30], ['{:02d}:{:02d}'.format(*divmod(420 + i, 60)) for i in range(len(problem_instance.T))][::30])
  plt.yticks([0, problem_instance.soc_max], ["SoC Low", "SoC Max"])
  plt.ylabel("SoC in %")
  plt.title("State of Charge of Fleet of eVTOLs over one day of service.")
  plt.legend()
  plt.savefig(f'/Users/mehdi/ATCP/test-expe/src/logs_results/Electric_plots/Charge_SoC_eVTOLs_SLS_{name}.png')



# @njit(inline="always")
def clear_unvisited_nodes(sol: object, problem_instance: ProblemInstance):
  """ Clear attribute of unvisited nodes in sol.

      Should not be used intensively, but only for debugging purposes.
      The attributes will be handled incrementally, after each move.

      Args:
            sol: solution instance to be cleared.
      Returns:
            None

      Updates:
            sol is updated in place after running this.

  """
  for r in range(problem_instance.n_requests):
    if sol.assignement[r] == -1:
      #unserved demand should have all zeros attributes
      clear_node(r, v, sol)
  return None

@njit(inline="always")
def find_insertion_slot(r:int, v: int, sol: object, problem_instance: object):
  """ Finds predecessor and successor requests in route of v, where r could be inserted.

      For an original path .. -> predecessor -> successor -> .., r would be inserted as
      .. -> predecessor -> r -> successor -> ..
      Insertion is feasible if of the added arcs are feasible.

      Args:
          r: id of request to be inserted
          v: id of aircraft
          sol: solution object

      Returns:
          predecessor: int, id of predecessor request
          successor: int, id of successor request
          feas: bool, insertion feasibility status
  """
  # print("insertion")
  # print(r, v)
  predecessor = int(problem_instance.n_requests + v)
  successor = sol.get_succ(predecessor, v)
  # print(predecessor, successor)
  if successor < 0:
    feas = problem_instance.time_compatible[predecessor, r] == 1
    return predecessor, successor, feas
  while r > successor and successor >= 0:
    predecessor = successor
    successor = sol.get_succ(predecessor, v)
  feas = problem_instance.time_compatible[predecessor, r] == 1
  if successor >= 0:
    feas = feas and problem_instance.time_compatible[r, successor] == 1
  # print("Exit slot")
  return predecessor, successor, feas

@njit(inline="always")
def clear_node(r: int, v: int, sol: object):
  """ Clear all attribute related to node r for v in sol.

      Clears: - charging times before/after
              - violation before/after
              - energy bought before/after
              - energy levels before/after

      Args:
            r: request id
            v: aircraft id
            sol: solution instance

      Returns:
            None

      Sol is modified in place.
  """
  sol.commit_new_charge_time_after(r, v, 0.)
  sol.commit_new_charge_time_before(r, v, 0.)
  sol.commit_new_energy_after(r, v, 0.)
  sol.commit_new_energy_before(r, v, 0.)
  sol.commit_new_energy_bought_after(r, v, 0.)
  sol.commit_new_energy_bought_before(r, v, 0.)
  sol.commit_new_violation_after(r, v, 0.)
  sol.commit_new_violation_before(r, v, 0.)
  sol.commit_new_assignment(r, -1.)
  sol.commit_new_succ(r, v, -1.)
  sol.commit_new_pred(r, v, -1.)
  return None

def compute_service_coef(problem_instance):
  """ Compute serice penalty coef """
  fc = compute_fastcharge_coef(problem_instance)
  return 9.321 * (fc + np.max(problem_instance.costs[problem_instance.costs < 10e4]))


def compute_fastcharge_coef(problem_instance):
  """ Cmputes fast charge penalty coef"""
  max_routing = np.max(problem_instance.costs[problem_instance.costs < 10e4])
  max_landing_fee = np.max(problem_instance.skyports[:, 1])
  max_electricity = 2 * problem_instance.soc_max * problem_instance.pe
  max_flying_time = np.max(problem_instance.time)
  max_vtol_cost = np.max(problem_instance.aircraft[:, 1])
  pen_fast = max_routing + max_landing_fee + max_electricity + max_flying_time * max_vtol_cost
  return pen_fast




@njit(inline="always")
def aggregate_diff(diff_routing: float,
                   diff_bought: float,
                   diff_viol: float,
                   problem_instance: object,
                   is_insertion: bool = False,
                   is_removal: bool = False,
                   diff_fast_charge: int = 0,
                   penalty: float = 0.):
  """ Returns the aggregation of differential.

      Individual differential are aggregated with their respective
      weights.

      Args:
          diff_routing: routing cost differential
          diff_bought: energy bought differential
          diff_viol: violation differential
          problem_instance: problem instance object
          is_insertion: true iif the diff aggregated is an insertion
                        will add the bonus for serving a demand to the diff.
                        Otherwise it will add the penalty for removing one.
          penalty: penalty for violation

      Returns:
          agg_diff: float, total differential

  """
  add = -1 if is_insertion else 0
  rm = 1 if is_removal else 0
  return diff_routing + diff_bought * problem_instance.pe + diff_viol * penalty + rm * problem_instance.lbda_u + add * problem_instance.lbda_u + diff_fast_charge * problem_instance.lbda_f


@njit(inline="always", parallel=False)
def golden_section_search(f,
        a,
        b,
        sol: object,
        r: int,
        v: int,
        problem_instance: object,
        move: int,
        tol: float = 1,
        penalty: float = 0,
        invphi: float = 0.618,
        invphi2: float = 0.382):
  """ Golden-section search, finding a local minimum for f.

        Given a function f, this function finds a local minima
        on the interval [a, b].

        The principle works for any function f but in this particular
        function, f is expected to take as argument sol, r, v and
        problem_instance. f should either be evaluate_charging_move_before / after.

        The idea of the optim is to start with a search interval [a, b], knowing f(a)
        f(b), selecting two points c and d, with c < d, in between and compare f(c)
        and f(d). Then comparing f(c) and f(d) we truncate part of the initial interval
        [a, b] :
        either changing a or b and the process is repeated. The location of c and d
        is chosen such that at iteration (or recursive call) t+1, one of the bounds of
        the interval is c(t) or d(t), thereby requiring only one function evaluation
        at each call instead of 2. The right truncation parameter can be derived
        analytically and is given a default value for inv_ratio. This value should
        NOT be changed.

        Args:
            f: function to minimize
            a: lower bound of the search interval
            b: upper bound of the search interval
            sol: solution object
            r: request id involved
            v: aircraft id involved
            problem_instance: problem
            move: move counter
            tol: width under which search stops.
            penalty: soc violation penalty
            inv_phi: inverse of gold number
            inv_phi2: inverse of gold number squared

        Returns:
            float: point at which f is minimized
            float: value of f at the local minimum found
            move: updated move counter
  """
  width = b - a
  if width <= tol:
    return 0., 0., move
  # Required steps to achieve tolerance
  c = a + invphi2 * width
  d = a + invphi * width

  db, dv, df = f(sol,
                      c,
                      v,
                      r,
                      problem_instance)
  yc = aggregate_diff(0,
                      db,
                      dv,
                      problem_instance,
                      diff_fast_charge=df,
                      penalty=penalty)
  db, dv, df = f(sol,
                    d,
                    v,
                    r,
                    problem_instance)
  yd = aggregate_diff(0,
                      db,
                      dv,
                      problem_instance,
                      diff_fast_charge=df,
                      penalty=penalty)
  n = int(np.ceil(np.log(tol / width) / np.log(invphi)))
  move += n
  for k in range(n-1):
    if yc < yd:
      b = d
      d = c
      yd = yc
      width = invphi * width
      c = a + invphi2 * width
      db, dv, df = f(sol,
                      c,
                      v,
                      r,
                      problem_instance)
      yc = aggregate_diff(0,
                          db,
                          dv,
                          problem_instance,
                          diff_fast_charge=df,
                          penalty=penalty)

    else:
      a = c
      c = d
      yc = yd
      width = invphi * width
      d = a + invphi * width
      db, dv, df = f(sol,
                      d,
                      v,
                      r,
                      problem_instance)
      yd = aggregate_diff(0,
                          db,
                          dv,
                          problem_instance,
                          diff_fast_charge=df,
                          penalty=penalty)

  if yc < yd:
    return c, yc, move
  else:
    return d, yd, move


@njit(inline="always")
def get_start_point(v: int, problem_instance: object):
  """ Get the starting point of an aircraft v

      Args:
          v: aircraft id
          problem_instance: ProblemInstance()

      Returns:
          int, skyport id
  """
  return problem_instance.requests[problem_instance.n_requests + v, 0]

@njit(inline="always")
def repair_rotation(sol: object, problem_instance: object):
  """ Repairs in place a rotation.

      When rotating routes between aircraft, the beginning of routes
      might not be feasible, this function repairs the eventual infeasibility
      and modifies sol in place.

      Args:
          sol: solution instance
          problem_instance: problem

      Returns:
          None
  """
  for v in range(problem_instance.n_aircraft):
    v_start = get_start_point(v, problem_instance)
    succ = sol.get_succ(v_start, v)
    if succ < 0:
      #route is empty
      continue
    feas = feasible_connection(v_start, succ, problem_instance.time_compatible)
    while not (feas):
      #-- clearing unfeasible node
      clear_node(succ, v, sol)
      sol.commit_new_assignment(succ, -1)
      # moving to next
      succ = sol.get_succ(succ, v)
      feas = feasible_connection(v_start, succ, problem_instance.time_compatible)
    sol.commit_new_succ(v_start, v, succ)
    sol.commit_new_pred(succ, v, v_start)
  # Repairing done
  return None

# @njit(inline="always")
def rotate_attributes_old(sol: object, order: int, problem_instance: object):
  """ Rotates solution attributes between aircraft.

      Rotating attributes means exchanging attributes between
      aircraft in a circular manner. Results in swapping entire
      routes.
      Assignements and Charging times are rotated and the rest
      of the attributes are reset to 0 since they are determined
      by the two first decision variables.

      Args:
          sol: solution instance
          order: order of the rotation

      Returns:
          None

      Sol is modified IN PLACE.
  """
  sol.assignementMap = np.roll(sol.assignementMap, shift=order, axis=2)
  sol.assignementMap[problem_instance.n_requests:, :, :] = np.roll(sol.assignementMap[problem_instance.n_requests:, :, :], shift=order, axis=0)
  sol.charging_times = np.roll(sol.charging_times, shift=order, axis=1)
  sol.energy_bought[:, :, :] = 0
  sol.energy_levels[:, :, :] = 0
  sol.violation[:,:,:] = 0
  mask = sol.assignement[:problem_instance.n_requests] >= 0
  sol.assignement[:problem_instance.n_requests][mask] = (sol.assignement[:problem_instance.n_requests][mask] + order) % problem_instance.n_aircraft
  return None

@njit(inline="always")
def rotate_attributes(sol: object, order: int, problem_instance: object):
  """ Rotates solution attributes between aircraft.

      Rotating attributes means exchanging attributes between
      aircraft in a circular manner. Results in swapping entire
      routes.
      Assignements and Charging times are rotated and the rest
      of the attributes are reset to 0 since they are determined
      by the two first decision variables.

      Args:
          sol: solution instance
          order: order of the rotation

      Returns:
          None

      Sol is modified IN PLACE.
  """
  rolled_charge_times, rolled_asignmap = roll_custom(sol, order, problem_instance)
  sol.assignementMap = rolled_asignmap
  sol.charging_times = rolled_charge_times
  sol.energy_bought[:, :, :] = 0
  sol.energy_levels[:, :, :] = 0
  sol.violation[:,:,:] = 0
  mask = sol.assignement[:problem_instance.n_requests] >= 0
  sol.assignement[:problem_instance.n_requests][mask] = (sol.assignement[:problem_instance.n_requests][mask] + order) % problem_instance.n_aircraft
  return None

@njit(inline="always")
def roll_custom(sol: object, order: int, problem_instance: object):
  ids = np.roll(problem_instance.ids_aicraft, order)
  rolled_assignementMap = np.zeros(shape=sol.assignementMap.shape) - 1
  rolled_charging_times = np.zeros(shape=sol.charging_times.shape)
  for i in range(problem_instance.n_aircraft):
    rolled_charging_times[:,i,:] = sol.charging_times[:,ids[i],:]
    rolled_assignementMap[:problem_instance.n_requests,:, i] = sol.assignementMap[:problem_instance.n_requests,:, ids[i]]
  # for i in range(problem_instance.n_aircraft):
    rolled_assignementMap[problem_instance.n_requests + i, 1, i] = sol.assignementMap[problem_instance.n_requests + ids[i], 1, ids[i]]
  return rolled_charging_times, rolled_assignementMap


@njit(inline="always")
def reset_costs(sol: object):
  """ Reset all cost of sol """
  sol.cost = 0.
  sol.electricity_cost = 0.
  sol.routing_cost = 0.
  sol.violation_tot = 0.
  return None



def commit_random_solution(sol: object, problem_instance: object):
  """ Commits a random solution to sol.

      Sol is modified in place to contain a random solution
      that is valid but does not respect the soc constraint.

      Args:
          sol: solution object
          problem_instance: problem

      Returns:
          None
  """
  np.random.shuffle(problem_instance.aircraft_ids)
  np.random.shuffle(problem_instance.requests_ids)
  # print("Comit random")
  #iterate over aircraft in random order
  for v in problem_instance.aircraft_ids:
    # print(f"Aicraft {v}")
    #insert requests as long as possible
    for r in problem_instance.requests_ids:
      # print(f"Request {r}")
      if sol.get_assigned_aircraft(r) < 0:
        pred, succ, feas = find_insertion_slot(r, v, sol, problem_instance)
        if feas:
          sol.commit_new_assignment(r, v)
          if succ >= 0:
            sol.commit_new_succ(r, v, succ)
            sol.commit_new_pred(succ, v, r)
          sol.commit_new_pred(r, v, pred)
          sol.commit_new_succ(pred, v, r)
  return None


def generate_mzn(problem_instance, s ,v, r, tag:str = ""):
  """  Generate data files containing data for a problem instance.

      Args:


      Returns:
            None

      Saving files to //src/log_results/instances_data/[FILENAME].dzn
  """
  info = ["aircraft", "requests", "energy", "time", "costs", "time_compatible"]
  exclude = ["v", "r", "s"]
  FILENAME = f"src/logs_results/instances_data/Model_BENCHMARK_{s}_{v}_{r}{tag}.mzn"
  attr = list(filter(lambda a: not a.startswith('__'), dir(problem_instance)))
  sheet = open(FILENAME, 'w')
  to_write = []
  for a in attr:
    if a in info:
      name = f"{a}_info_1d ="
      to_write.append(f"{name} {list(getattr(problem_instance, a).flatten())};\n")
      to_write.append(f"int: n_{a} = {getattr(problem_instance, a).shape[0]};\n")
    elif a not in exclude and isinstance(getattr(problem_instance, a), int):
      if a not in ["n_aircraft", "n_requests"]:
        name = f"int: {a} ="
        to_write.append(f"{name} {getattr(problem_instance, a)};\n")
    elif isinstance(getattr(problem_instance, a), float):
      name = f"float: {a} ="
      to_write.append(f"{name} {getattr(problem_instance, a)};\n")
  # defaut params to get into the program
  to_write += f"int: n_skyports = {s};\n"
  to_write += "int: source_location = 1;\n"
  to_write += "int: dest_location = 2;\n"
  to_write += "int: source_time = 3;\n"
  to_write += "int: dest_time = 4;\n"
  to_write += "float: domain_charge_float = 100.0;\n"
  to_write += f"float: pen_unserved = {problem_instance.lbda_u};\n"
  to_write += f"float: pen_fast = {problem_instance.lbda_f};\n"
  to_write += f"int: soc_high = {int(problem_instance.soc_max)};\n"
  sheet.writelines(to_write)
  #--add model constraints - _small.mzn contains model where constraints are instanciated
  # only when it is useful (on feasible arcs) - This reduces the compilation time
  # and hopefully makes it possible to run larger instances without blowing up memory..
  with open("src/electric/model_constraints_small.mzn") as infile:
    sheet.write(infile.read())
  sheet.close()
  print("File written.")
  corresponding_output = f"src/logs_results/Gurobi_output/output_{s}_{v}_{r}"
  return FILENAME, corresponding_output
  # print("File written to //src/log_results/instances_data/")


@dataclass
class MinizincSolution:
  y: np.ndarray = None
  ta: np.ndarray = None
  tb: np.ndarray = None
  sa: np.ndarray = None
  sb: np.ndarray = None
  ba: np.ndarray = None
  bb: np.ndarray = None
  b: np.ndarray = None
  e_before: np.ndarray = None
  e_after: np.ndarray = None
  unserved: int = None
  fast_charges: int = None
  time: float = None


  def get_op_cost(self, problem_instance):
    """ """
    routing_cost = 0
    electricity_cost = 0
    for v in range(self.y.shape[2]):
      for i in range(self.y.shape[0]):
        for j in range(self.y.shape[1]):
          if self.y[i, j, v]:
            routing_cost += problem_instance.costs[i, j]
            electricity_cost += self.b[i, j, v] * problem_instance.pe
    return routing_cost, electricity_cost

  def get_nb_charge(self):
    """ """
    nb_charging = 0
    for v in range(self.y.shape[2]):
      for i in range(self.y.shape[0]):
        for j in range(self.y.shape[1]):
          if self.y[i, j, v]:
            if self.ta[i, j, v] != 0:
              nb_charging += 1
            if self.tb[i, j, v] != 0:
              nb_charging += 1
    return nb_charging



@dataclass
class PoolingSolution:
  demands: np.ndarray = None
  x: np.ndarray = None
  y: np.ndarray = None
  f: np.ndarray = None
  a: np.ndarray = None
  usage: int = None
  overall_waiting: float = None



def process_minizinc_output(output_file: str,
                            flat_time: float,
                            max_time: float,
                            pool: bool = False):
  """ Process decision var returned by gurobi
      to compute electricity and routing cost.
      For debugging and checks.
  """
  fileo = open(output_file, "r")
  content = fileo.read()
  solutions = content.split("----------")
  all_solutions = []
  #--- Get final time after sol proven optimal if happenned.
  if "===" not in solutions[-1]:
    time_final = max_time + flat_time
  else:
    time_final = solutions[-1][solutions[-1].find(": ") + 1:solutions[-1].find(" s")]
  #---
  for sol in solutions[:-1]:
    if pool:
      ObjSol = PoolingSolution()
    else:
      ObjSol = MinizincSolution()
    elements = sol.split(";")
    for ele in elements:
      if "elapsed" in ele:
        time_solve = ele[ele.find(": ") + 1:ele.find(" s")]
        ObjSol.time = float(time_solve) - flat_time
        break
      if "=" not in ele:
        continue
      attr, value = ele.split(" = ")
      attr = attr.replace("\n", "")
      if "array" in value:
        value = value[value.find("(") + 1:value.find(")")]
        shapestr = value[:value.find("[")]
        m = re.findall('..(\d+),', shapestr, re.IGNORECASE)
        shape = tuple([int(x) + 1 for x in m])
        tab = value[value.find("[")+1:value.find("]")].split(",")
        tab = [v for v in tab if v not in ["[", "]"]]
        if attr in ["y", "sa", "sb", "x"]:
          tab = [v.replace(" ", "") for v in tab]
          table = np.array(tab) == "true"
        else:
          tab = [float(v) for v in tab]
          table = np.array(tab)
        table = table.reshape(shape)
        setattr(ObjSol, attr, table)
      else:
        setattr(ObjSol, attr, float(value))
    all_solutions.append(ObjSol)
  return all_solutions, float(time_final)



def init_vns(s,
             v,
             r,
             problem_instance,
             commit_greedy_cost,
             commit_propagation_new_energy,
             SolutionSls):
  """ Initiate necessary solution instance prior to running VNS.

  """
  ObjSol = SolutionSls
  solution = initiate_solution(s, v, r, problem_instance.soc_max, ObjSol)
  commit_random_solution(solution, problem_instance)
  best_sol = initiate_solution(s, v, r, problem_instance.soc_max, ObjSol)
  staging_sol = initiate_solution(s, v, r, problem_instance.soc_max, ObjSol)
  random_sol = initiate_solution(s, v, r, problem_instance.soc_max, ObjSol)

  for v in problem_instance.aircraft_ids:
    start_v = problem_instance.n_requests + v
    commit_propagation_new_energy(v,
                                  start_v,
                                  solution,
                                  problem_instance.soc_max,
                                  problem_instance)

  commit_greedy_cost(solution, problem_instance)
  #--
  return solution, best_sol, staging_sol, random_sol


def count_all_charges_vns(sol, problem_instance):
  """ Counts all charging operation in sol of VNS."""
  nb_charging = 0
  for v in problem_instance.aircraft_ids:
    predecessor = problem_instance.n_requests + v
    successor = sol.get_succ(predecessor, v)
    while predecessor >= 0:
      if sol.get_charging_time_after(predecessor, v) != 0:
        nb_charging += 1
      if successor < 0:
        break
      if sol.get_charging_time_before(successor, v) != 0:
        nb_charging += 1
      predecessor = successor
      if predecessor < 0:
        continue
      successor = sol.get_succ(predecessor, v)
  return nb_charging

