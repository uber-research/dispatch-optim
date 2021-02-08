####
# Module to evaluate candidate solutions and evaluate moves.
#
####
import numba
from numba import njit, jit
import logging
import numpy as np
import sys
sys.path.append("../")
from utils_electric import (
    find_insertion_slot,
    clear_node,
    get_start_point,
    feasible_connection
)


@njit(inline="always")
def time_to_energy(t: float,
                   problem_instance: object):
  """ This function converts a charging time into a delta of SoC units.
      With the current assumptions, the function is simple but could be
      arbitrarily complex.
      Since we have two modes only, negative (resp. positive) values
      are interpreted as slow (resp. fast) charge time.

      Args:
          t: charging time in minutes
          gamma_s: slow charging rate, in SoC units per minute
          gamma_f: fast charging rate, in SoC units per minute

      Returns:
          Additionnal SoC units gained after charging t minutes.
  """
  if t < 0:
    return abs(t) * problem_instance.gamma_s
  return t * problem_instance.gamma_f


@njit(inline="always")
def compute_violation(energy_level: float, min_soc: float):
  """ Compute violation degree of this level of energy.

      Args:
          energy_level: SoC units
          min_soc: minimum SoC for takeoff

      Returns:
          float, violation degree
  """
  return max(0, min_soc - energy_level)

@njit(inline="always")
def compute_diff_viol_before(sol: object,
                             r: int,
                             v: int,
                             e_lvl: float,
                             min_soc: float):
  """ Differential in violation before r for v.

      Computing the differential in violation before request r for aircraft v.
      Given that energy level is e_lvl. The differential is taken w.r.t
      solution object sol.

      Args:
          sol: SolutionSls instance
          r: request id
          v: aircraft id
          e_lvl: energy level at r- for v
          min_soc: minimum soc for takeoff

      Returns:
          diff_viol_before: float, differential in violation.
  """
  old_viol_before = sol.get_violation_before(r, v)
  diff_viol_before = compute_violation(e_lvl, min_soc) - old_viol_before
  return diff_viol_before

@njit(inline="always")
def compute_diff_viol_after(sol: object,
                             r: int,
                             v: int,
                             e_lvl: float,
                             min_soc: float,
                             energy_dh: float):
  """ Differential in violation after r for v.

      Computing the differential in violation after request r for aircraft v.
      Given that energy level is e_lvl. The differential is taken w.r.t
      solution object sol.

      Args:
          sol: SolutionSls instance
          r: request id
          v: aircraft id
          e_lvl: energy level at r- of v at deadhead time (after charging)
          min_soc: minimum soc for takeoff
          energy_dh: energy required for the deadhead between r and its successor.

      Returns:
          diff_viol_after: float, differential in violation.
  """
  old_viol_after = sol.get_violation_after(r, v)
  new_violation_after = 0
  if energy_dh > 0:
    new_violation_after += compute_violation(e_lvl, min_soc)
  diff_viol_after = new_violation_after - old_viol_after
  return diff_viol_after

@njit(inline="always")
def compute_diff_bought_after(sol: object,
                              r: int,
                              v: int,
                              e_lvl: float,
                              problem_instance: object):
  """ Differential in energy bought after r for v.

      Computing the differential in energy bought after request r for aircraft v.
      Given that energy level is e_lvl. The differential is taken w.r.t
      solution object sol.

      Args:
          sol: SolutionSls instance
          r: request id
          v: aircraft id
          e_lvl: energy level at r- for v
          soc_max: max battery soc

      Returns:
          diff_bought_after: float, differential in energy bought.
          new_bought_after: float, new energy bought after r by v.
  """
  ta = sol.get_charging_time_after(r, v)
  old_bought_after = sol.get_energy_bought_after(r, v)
  new_bought_after = min(problem_instance.soc_max - e_lvl, time_to_energy(ta, problem_instance))
  diff_bought_after = new_bought_after - old_bought_after
  return diff_bought_after, new_bought_after

@njit(inline="always")
def compute_diff_bought_before(sol: object,
                              r: int,
                              v: int,
                              e_lvl: float,
                              problem_instance: object):
  """ Differential in energy bought before r for v.

      Computing the differential in energy bought after request r for aircraft v.
      Given that energy level is e_lvl. The differential is taken w.r.t
      solution object sol.

      Args:
          sol: SolutionSls instance
          r: request id
          v: aircraft id
          e_lvl: energy level at r- for v
          soc_max: max battery soc
          successor: successor of r in route of v.

      Returns:
          diff_bought_before: float, differential in energy bought.
          new_bought_before: float, new energy bought before r by v.
  """
  tb = sol.get_charging_time_before(r, v)
  old_bought_before = sol.get_energy_bought_before(r, v)
  new_bought_before = min(problem_instance.soc_max - e_lvl, time_to_energy(tb, problem_instance))
  diff_bought_before = new_bought_before - old_bought_before
  return diff_bought_before, new_bought_before

@njit(inline="always")
def propagate_energy_one_step(r: int,
                              successor: int,
                              v: int,
                              sol: object,
                              incoming_energy_level: float,
                              problem_instance: object):
  """ Propagates one step ahead the effect of changing the energy level at before r for v
      until before next request.
      Propagation between r and its successors, that is:
        - energy violation at takeoff from r- to r+ (i.e. serving r)
        - energy violation when deadheading from r+ to succ-
        - energy bought after r (i.e. at r+, before the deadhead)
        - energy bought before succ (i.e. at succ- before serving succ)

      Executed iteratively, it will propagates energy levels, violations and quantities purchased
      through the routes of an aircraft.

      Args:
          r: request id
          v: aircraft id
          min_soc: min soc
          sol: SolutionSls object
          energy_dep: new energy value input at before r for v
      Returns:
          diff_viol_before: energy violation differential
          energy_next: energy at next step
  """
  energy_dh = problem_instance.get_connection_energy(r, successor)
  energy_service = problem_instance.get_service_energy(r)
  diff_viol_before = compute_diff_viol_before(sol,
                                              r,
                                              v,
                                              incoming_energy_level,
                                              problem_instance.min_soc)
  energy_lvl = incoming_energy_level - energy_service
  if successor < 0:
    #end of route is reached stop after the service.
    return diff_viol_before, 0., 0., 0., energy_lvl
  diff_bought_after, new_bought_after = compute_diff_bought_after(sol,
                                                                  r,
                                                                  v,
                                                                  energy_lvl,
                                                                  problem_instance)

  diff_viol_after = compute_diff_viol_after(sol,
                                            r,
                                            v,
                                            energy_lvl + new_bought_after,
                                            problem_instance.min_soc,
                                            energy_dh)
  diff_bought_before, new_bought_before = compute_diff_bought_before(sol,
                                                                      successor,
                                                                      v,
                                                                      energy_lvl + new_bought_after - energy_dh,
                                                                      problem_instance)

  new_energy = energy_lvl + new_bought_after - energy_dh + new_bought_before
  return diff_viol_before, diff_viol_after, diff_bought_after, diff_bought_before, new_energy


@njit(inline="always")
def propagate_new_energy(v: int,
                         r: int,
                         sol: object,
                         start_energy: float,
                         problem_instance: object,
                         propagate_energy_one_step=propagate_energy_one_step):
  """ Full propagation of the energy levels and violations of vtol v from r.

      This function propagates the energy levels and violations of vtol v starting from request
      beggining of r and stops whenever the delta is 0 (does not change from current solution).
      Used whenever wanting to evaluate a move that will impact the energy level at r.
      Every sol parameter should not have been changed after r.
      Currently it starts from the input request, it is stopped
      whenever it is certain that futur differential will be zeros : when the energy level at
      before successor is the same as it was before.


      Args:
          v: id of aircraft
          r: id of requests
          sol: SolutionSls object
          start_energy: battery soc at before r for v
          problem_instance: problem object
          stop: request at which propagation should stop, -1 if none.

      Returns:
          diff_electricity: float, new energy violation after propagation
          diff_viol: float, new energy violation after propagation
  """
  diff_viol = 0
  diff_bought = 0
  incoming_energy_level = start_energy
  current = int(r)
  successor = sol.get_succ(current, v)
  while current >= 0:
    #iterate until reaching end of route.
    dv_before, dv_after, db_after, db_before, new_energy = propagate_energy_one_step(current,
                                                                                      successor,
                                                                                      v,
                                                                                      sol,
                                                                                      incoming_energy_level,
                                                                                      problem_instance)

    incoming_energy_level = new_energy
    if successor < 0:
      diff_viol += dv_before
      diff_bought += db_before
    else:
      diff_viol += dv_before + dv_after
      diff_bought += db_before + db_after
    if new_energy == sol.get_energy_before(successor, v):
      break

    current = successor
    if current < 0:
      continue
    successor = sol.get_succ(current, v)
  return diff_bought, diff_viol

@njit(inline="always")
def commit_propagation_new_energy(v: int,
                                  r: int,
                                  sol: object,
                                  start_energy: float,
                                  problem_instance: object,
                                  penalty: float = 0.):
  """ Full COMMITED propagation of the energy levels and violations of vtol v from r.

      This function propagates the energy levels and violations of vtol v starting from request
      beggining of r and stops whenever the delta is 0 (does not change from current solution).
      Used whenever wanting to evaluate a move that will impact the energy level at r.
      Every sol parameter should not have been changed after r[at]. The propagation is
      also commited to sol. Therefore the sol instance is modified after executing this function.

      Args:
          v: id of aircraft
          r: id of requests
          sol: SolutionSls object
          at: either 0 or 1 indicating if the starting point is after (0) or before (1) r
          ea: new energy value to propagate from
          min_soc: minimum soc for takeoff

      Returns:
          diff_electricity: float, new energy violation after propagation
          diff_viol: float, new energy violation after propagation
  """
  incoming_energy_level = start_energy
  current = int(r)
  successor = sol.get_succ(current, v)
  #-- Assigning lagged variables to default values for
  # numba variables declaration.
  current_lag = -1
  successor_lag = -1
  dv_before_lag = 0.
  dv_after_lag = 0.
  db_after_lag = 0.
  db_before_lag = 0.
  incoming_energy_level_lag = 0.
  new_energy_lag = 0.
  while current >= 0:
    #iterate until reaching end of route.
    dv_before, dv_after, db_after, db_before, new_energy = propagate_energy_one_step(current,
                                                                                      successor,
                                                                                      v,
                                                                                      sol,
                                                                                      incoming_energy_level,
                                                                                      problem_instance)

    # Commiting values with lag to avoig endogeneous behavior.

    if current_lag >= 0:
      sol.commit_new_energy_bought_after(current_lag, v, sol.get_energy_bought_after(current_lag, v) + db_after_lag)
      sol.commit_new_energy_bought_before(successor_lag, v, sol.get_energy_bought_before(successor_lag, v) + db_before_lag)
      sol.commit_new_violation_before(current_lag, v, sol.get_violation_before(current_lag, v) + dv_before_lag)
      sol.commit_new_violation_after(current_lag, v, sol.get_violation_after(current_lag, v) + dv_after_lag)
      sol.commit_new_energy_before(current_lag, v, incoming_energy_level_lag)
      energy_service = problem_instance.get_service_energy(current_lag)
      energy_dh = problem_instance.get_connection_energy(current_lag, successor_lag)
      sol.commit_new_energy_after(current_lag, v, incoming_energy_level_lag - energy_service)

    #-- lagged values update
    current_lag = current
    successor_lag = successor
    dv_before_lag, dv_after_lag, db_after_lag, db_before_lag, incoming_energy_level_lag, new_energy_lag = dv_before, dv_after, db_after, db_before, incoming_energy_level, new_energy
    # Comitting last request of route
    if successor < 0:
      sol.commit_new_violation_before(current, v, sol.get_violation_before(current, v) + dv_before)
      sol.commit_new_violation_after(current, v, 0)
      sol.commit_new_energy_before(current, v, incoming_energy_level)
      sol.commit_new_energy_after(current, v, new_energy)
      sol.commit_new_energy_bought_after(current, v, 0)
      sol.commit_new_charge_time_after(current, v, 0)
    #--
    incoming_energy_level = new_energy
    current = successor
    if successor < 0:
      sol.violation_tot += dv_before
      sol.electricity_cost += db_before * problem_instance.pe
      sol.cost += (db_before) * problem_instance.pe + penalty * (dv_before)
    else:
      sol.violation_tot += dv_before + dv_after
      sol.electricity_cost += (db_before + db_after) * problem_instance.pe
      sol.cost += (db_before + db_after) * problem_instance.pe + penalty * (dv_before + dv_after)
    if current < 0:
      continue
    successor = sol.get_succ(current, v)
  return None

@njit(inline="always")
def compute_greedy_cost(sol: object, problem_instance: object, penalty: float = 0.):
  """ Compute total cost of a solution greedily.

      Args:
          sol: solution object
          problem_instance: instance of the problem

      Returns:
          total_cost: float, total cost of solution including penalties
          routing_cost: float, cost of routing part only
          electricity cost: float, cost of the electricity bought only
          unserved_count: int, number of unserved demands
          violation: float, total violation degree of solution
          fast_charge_count: int, number of fast charges used.
  """
  total_cost = 0.
  routing_cost = 0.
  electricity_cost = 0.
  unserved_count = problem_instance.n_requests
  violation = 0.
  fast_charge_count = 0
  for v in range(problem_instance.n_aircraft):
    current = int(problem_instance.n_requests + v)
    successor = sol.get_succ(current, v)
    energy_lvl = problem_instance.soc_max
    while successor >= 0:
      #-- Unfeasibility safeguard
      if not (feasible_connection(current, successor, problem_instance.time_compatible)):
        print(current, successor)
        raise ValueError("Connection attempted but not feasible !")
      #--
      # print(f"Greedy {current} -> {successor}. Violation {violation}")
      routing_cost += problem_instance.costs[current, successor]
      ta = sol.get_charging_time_after(current, v)
      if ta > 0:
        fast_charge_count += 1
      energy_lvl -= problem_instance.get_service_energy(current)
      e_bought = sol.get_energy_bought_after(current, v) #min(problem_instance.soc_max - energy_lvl, time_to_energy(ta, problem_instance))
      electricity_cost += problem_instance.pe * e_bought
      dh_energy = problem_instance.get_connection_energy(current, successor)
      energy_lvl = energy_lvl + e_bought
      if dh_energy > 0:
        violation += compute_violation(energy_lvl, problem_instance.min_soc)
      energy_lvl = energy_lvl - dh_energy
      tb = sol.get_charging_time_before(successor, v)
      if tb > 0:
        fast_charge_count += 1
      e_bought = sol.get_energy_bought_before(successor, v) #min(problem_instance.soc_max - energy_lvl, time_to_energy(tb, problem_instance))
      energy_lvl += e_bought
      # if successor == 20:
      #   print(f"Energy before {successor} is {energy_lvl}")
      electricity_cost += problem_instance.pe * e_bought
      violation += compute_violation(energy_lvl, problem_instance.min_soc)
      current = successor
      unserved_count -= 1
      if current < 0:
        continue
      successor = sol.get_succ(current, v)
  # aggregating
  total_cost = routing_cost + electricity_cost + unserved_count * problem_instance.lbda_u + fast_charge_count * problem_instance.lbda_f + violation * penalty
  return total_cost, routing_cost, electricity_cost, violation, unserved_count, fast_charge_count

@njit(inline="always")
def commit_greedy_cost(sol: object, problem_instance: object, penalty: float = 0.):
  """ COMMIT greedy cost computation to sol

      Args:
          sol: solution object
          problem_instance: problem object

      Returns:
          None

      sol is modified in place when running this.
  """
  total_cost, routing_cost, electricity_cost, violation, unserved_count, fast_charge_count = compute_greedy_cost(sol, problem_instance, penalty = penalty)
  sol.cost = total_cost
  sol.routing_cost = routing_cost
  sol.electricity_cost = electricity_cost
  sol.violation_tot = violation
  sol.unserved_count = unserved_count
  sol.fast_charge_count = fast_charge_count
  return None

@njit(inline="always")
def evaluate_charging_move_before(sol: object,
                                  new_tb: float,
                                  v: int,
                                  r: int,
                                  problem_intance: object,
                                  propagate_new_energy=propagate_new_energy):
  """ Evaluate cost impact of changing a charging time before r for v.

      Assuming the only change in sol is the charging time before r for v, this
      will propagate the impact of this change on every variable that is impacted
      and return the resulting overall differential in cost.

      Args:
          sol: SolutionSls object
          new_tb: new charging time value
          v: id of aircraft
          r: id of requests
          problem_instance: problem object

      Returns:
          diff_viol: float, differential in violation implied by the change.
          diff_bought: float, differential in energy bought implied by the change.
  """
  #Clipping value to maximum gettable energy here.
  old_energy_bought_before = sol.get_energy_bought_before(r, v)
  energy_before_charge = sol.get_energy_before(r, v) - old_energy_bought_before
  new_bought_before = min(problem_intance.soc_max - energy_before_charge, time_to_energy(new_tb, problem_intance))
  diff_bought = new_bought_before - old_energy_bought_before
  new_energy_before = energy_before_charge + new_bought_before
  #-- Fast charge diff
  diff_fast = 0
  old_tb = sol.get_charging_time_before(r, v)
  diff_fast = 0
  if old_tb <= 0 and new_tb > 0:
    diff_fast += 1
  elif old_tb > 0 and new_tb <= 0:
    diff_fast -= 1
  # -- Now change is initialized at before r and we can propagate from there
  db, dv = propagate_new_energy(v,
                                r,
                                sol,
                                new_energy_before,
                                problem_intance)
  return diff_bought + db, dv, diff_fast


@njit(inline="always")
def evaluate_charging_move_after(sol: object,
                                  new_ta: float,
                                  v: int,
                                  r: int,
                                  problem_instance: object,
                                  propagate_new_energy=propagate_new_energy):
  """ Evaluate cost impact of changing a charging time after r for v.

      Assuming the only change in sol is the charging time before r for v, this
      will propagate the impact of this change on every variable that is impacted
      and return the resulting overall differential in cost.

      Args:
          sol: SolutionSls object
          new_ta: new charging time value
          v: id of aircraft
          r: id of requests
          problem_instance: instance of problem

      Returns:
          diff_viol: float, differential in violation implied by the change.
          diff_bought: float, differential in energy bought implied by the change.
  """
  successor = sol.get_succ(r, v)
  if successor < 0:
    return 0., 0., 0.
  # -- Clipping value to maximum gettable value here
  old_energy_bought_after = sol.get_energy_bought_after(r, v)
  new_bought_after = time_to_energy(new_ta, problem_instance)
  diff_bought = new_bought_after - old_energy_bought_after
  #-- Fast charge diff
  old_ta = sol.get_charging_time_after(r, v)
  diff_fast = 0
  if old_ta <= 0 and new_ta > 0:
    diff_fast += 1
  elif old_ta > 0 and new_ta <= 0:
    diff_fast -= 1
  #-- deadheading
  energy_lvl = sol.get_energy_after(r, v) + new_bought_after
  energy_dh = problem_instance.get_connection_energy(r, successor)
  diff_viol = compute_diff_viol_after(sol, r, v, energy_lvl, problem_instance.min_soc, energy_dh)
  energy_lvl -= energy_dh
  new_bought_before = min(problem_instance.soc_max - energy_lvl, sol.get_energy_bought_before(successor, v))
  energy_lvl += new_bought_before
  diff_bought += new_bought_before - sol.get_energy_bought_before(successor, v)
  # -- Now change is initialized at before r and we can propagate from there
  db, dv = propagate_new_energy(v,
                                successor,
                                sol,
                                energy_lvl,
                                problem_instance)
  return diff_bought + db, diff_viol + dv, diff_fast

@njit(inline="always")
def commit_charging_move_before(sol: object,
                                  new_tb: float,
                                  v: int,
                                  r: int,
                                  problem_instance: object,
                                  penalty: float = 0.,
                                  commit_propagation_new_energy=commit_propagation_new_energy):
  """ COMMITS cost impact of changing a charging time before r for v.

      Assuming the only change in sol is the charging time before r for v, this
      will propagate the impact of this change on every variable that is impacted
      and return the resulting overall differential in cost.

      Args:
          sol: SolutionSls object
          new_tb: new charging time value
          v: id of aircraft
          r: id of requests
          problem_instance: problem object

      Returns:
          diff_viol: float, differential in violation implied by the change.
          diff_bought: float, differential in energy bought implied by the change.
  """
  new_bought_before = time_to_energy(new_tb, problem_instance)
  diff_bought = new_bought_before - sol.get_energy_bought_before(r, v)
  new_energy_before = sol.get_energy_before(r, v) + diff_bought
  #-- Fast charge change
  old_tb = sol.get_charging_time_before(r, v)
  if old_tb <= 0. and new_tb > 0.:
    sol.cost += problem_instance.lbda_f
    sol.fast_charge_count += 1
  elif old_tb > 0. and new_tb <= 0.:
    sol.cost -= problem_instance.lbda_f
    sol.fast_charge_count -= 1
  sol.commit_new_charge_time_before(r, v, new_tb)
  sol.commit_new_energy_bought_before(r, v, new_bought_before)
  sol.electricity_cost += diff_bought * problem_instance.pe
  sol.cost += diff_bought * problem_instance.pe
  # -- Now change is initialized at before r and we can propagate from there
  commit_propagation_new_energy(v,
                                r,
                                sol,
                                new_energy_before,
                                problem_instance,
                                penalty = penalty)
  return None

@njit(inline="always")
def commit_charging_move_after(sol: object,
                                  new_ta: float,
                                  v: int,
                                  r: int,
                                  problem_instance: object,
                                  penalty: float = 0.,
                                  commit_propagation_new_energy=commit_propagation_new_energy):
  """ COMMITS cost impact of changing a charging time after r for v.

      Assuming the only change in sol is the charging time before r for v, this
      will propagate the impact of this change on every variable that is impacted
      and return the resulting overall differential in cost.

      Args:
          sol: SolutionSls object
          new_ta: new charging time value
          v: id of aircraft
          r: id of requests
          problem_instance: instance of problem

      Returns:
          diff_viol: float, differential in violation implied by the change.
          diff_bought: float, differential in energy bought implied by the change.
  """

  successor = sol.get_succ(r, v)
  if successor < 0:
    return None
  # -- Clipping value to maximum gettable value here
  # old_energy_bought_after = sol.get_energy_bought_after(r, v)
  energy_after = sol.get_energy_after(r, v)
  # new_bought_after = min(problem_instance.soc_max - energy_after, time_to_energy(new_ta, problem_instance))
  # diff_bought = new_bought_after - old_energy_bought_after
  # new_energy_after = old_energy_after + new_bought_after
  diff_bought = time_to_energy(new_ta, problem_instance) - sol.get_energy_bought_after(r, v)
  new_bought_after = time_to_energy(new_ta, problem_instance)
  # if new_bought_after + energy_after > 100. :
  #   print(new_ta, new_bought_after, energy_after)
  #   sys.exit
  #-- Fast charge change
  old_ta = sol.get_charging_time_after(r, v)
  if old_ta <= 0. and new_ta > 0.:
    sol.cost += problem_instance.lbda_f
    sol.fast_charge_count += 1
  elif old_ta > 0. and new_ta <= 0.:
    sol.cost -= problem_instance.lbda_f
    sol.fast_charge_count -= 1
  sol.commit_new_charge_time_after(r, v, new_ta)
  #-- deadheading
  # sol.commit_new_energy_after(r, v, energy_after)
  energy_dh = problem_instance.get_connection_energy(r, successor)
  sol.commit_new_energy_bought_after(r, v, new_bought_after)
  energy_lvl = energy_after + new_bought_after
  new_violation_after = 0
  if energy_dh > 0:
    new_violation_after += compute_violation(energy_lvl, problem_instance.min_soc)
  diff_viol = new_violation_after - sol.get_violation_after(r, v)
  sol.commit_new_violation_after(r, v, new_violation_after)
  energy_lvl -= problem_instance.get_connection_energy(r, successor)
  diff_bought_before, new_bought_before = compute_diff_bought_before(sol, successor, v, energy_lvl, problem_instance)
  energy_lvl += new_bought_before
  diff_bought += diff_bought_before
  sol.electricity_cost += diff_bought * problem_instance.pe
  sol.violation_tot += diff_viol
  sol.cost += diff_viol * penalty + diff_bought * problem_instance.pe
  sol.commit_new_energy_bought_before(successor, v, new_bought_before)
  # -- Now change is initialized at before r and we can propagate from there
  commit_propagation_new_energy(v,
                                successor,
                                sol,
                                energy_lvl,
                                problem_instance,
                                penalty = penalty)
  return None



@njit(inline="always")
def left_connection(pred: int,
                    r: int,
                    v: int,
                    sol: object,
                    problem_instance: object,
                    compute_diff_viol_after=compute_diff_viol_after):
  """ Computes differential in energy and violation for a new left connection.

      When inserting a request two connections are made : left and right.

      .. -> predecessor -> successor -> .., r would be inserted as
      .. -> predecessor -> r -> successor -> ..

      The left connection is predecessor -> r.

      Args:
          r: request id to be inserted
          v: aircraft id
          sol: solution object
          problem_instance: problem object

      Returns:
          db_left: float, differential in left energy bought
          dv_left: float, differential in left violation
          new_energy: float, new energy at r
  """
  energy_lvl = sol.get_energy_after(pred, v)
  energy_dh_r = problem_instance.get_connection_energy(pred, r)
  # Clipping the value of charging time in case too large for insertion.
  # idle_time = problem_instance.get_idle_time(pred, r)
  # ta = sol.get_charging_time_after(pred, v)
  # ta_clipped = min(ta, idle_time - problem_instance.delta)
  max_charge_time = problem_instance.get_max_charge_time(pred, r)
  ta = sol.get_charging_time_after(pred, v)
  sign = -1 if ta < 0 else 1
  ta_clipped = sign * min(abs(ta), max_charge_time)
  # Compute energy bought diff
  old_energy_bought_after = sol.get_energy_bought_after(pred, v)
  new_energy_bought_after = min(problem_instance.soc_max - energy_lvl, time_to_energy(ta_clipped, problem_instance))
  db_left = new_energy_bought_after - old_energy_bought_after
  # Violation diff
  dv_left = compute_diff_viol_after(sol,
                                    pred,
                                    v,
                                    energy_lvl + new_energy_bought_after,
                                    problem_instance.min_soc,
                                    energy_dh_r)
  # Energy after deadhead : before r
  new_energy = energy_lvl - energy_dh_r
  return db_left, dv_left, new_energy

@njit(inline="always")
def right_connection(energy_lvl: float,
                     succ: int,
                     r: int,
                     v: int,
                     sol: object,
                     problem_instance: object,
                     compute_diff_viol_after=compute_diff_viol_after):
  """ Computes differential in energy and violation for a new right connection.

      When inserting a request two connections are made : left and right.

      .. -> predecessor -> successor -> .., r would be inserted as
      .. -> predecessor -> r -> successor -> ..

      The right connection is r -> successor.

      Args:
          energy_lvl: incoming energy level
          r: request id to be inserted
          succ: request id of successor
          v: aircraft id
          sol: solution object
          problem_instance: problem object

      Returns:
          db_right: float, differential in right energy bought
          dv_right: float, differential in right violation
          new_energy: float, new energy at r
  """
  dv_right = compute_violation(energy_lvl, problem_instance.min_soc) # for serving r
  energy_lvl -= problem_instance.get_service_energy(r)
  energy_dh_succ = problem_instance.get_connection_energy(r, succ)
  # Violation when deadheading from r to succ
  dv_right += compute_diff_viol_after(sol,
                                        r,
                                        v,
                                        energy_lvl,
                                        problem_instance.min_soc,
                                        energy_dh_succ)
  energy_lvl -= energy_dh_succ
  # Clipping the value of charging time in case too large for insertion.
  # tb = sol.get_charging_time_before(succ, v)
  # idle_time = problem_instance.get_idle_time(r, succ)
  # tb_clipped = min(tb, idle_time - problem_instance.delta)
  tb = sol.get_charging_time_before(succ, v)
  max_charge_time = problem_instance.get_max_charge_time(r, succ)
  sign = -1 if tb < 0 else 1
  tb_clipped = sign * min(abs(tb), max_charge_time)
  # Differential in energy bought
  old_energy_bought_before = sol.get_energy_bought_before(succ, v)
  new_energy_bought_before = min(problem_instance.soc_max - energy_lvl, time_to_energy(tb_clipped, problem_instance))
  db_right = new_energy_bought_before - old_energy_bought_before
  # Setting new energy value
  energy_lvl += new_energy_bought_before
  return db_right, dv_right, energy_lvl

@njit(inline="always")
def commit_left_connection(pred: int, r: int, v: int, sol: object, problem_instance: object, penalty: float = 0.):
  """ Computes and COMMITS differential in energy and violation for a new left connection.

      When inserting a request two connections are made : left and right.

      .. -> predecessor -> successor -> .., r would be inserted as
      .. -> predecessor -> r -> successor -> ..

      The left connection is predecessor -> r.

      Args:
          r: request id to be inserted
          v: aircraft id
          sol: solution object
          problem_instance: problem object

      Returns:
          db_left: float, differential in left energy bought
          dv_left: float, differential in left violation
          new_energy: float, new energy at r

      sol is modified in place when calling this function.
  """
  energy_lvl = sol.get_energy_after(pred, v)
  energy_dh_r = problem_instance.get_connection_energy(pred, r)
  # Clipping the value of charging time in case too large for insertion.
  max_charge_time = problem_instance.get_max_charge_time(pred, r)
  ta = sol.get_charging_time_after(pred, v)
  sign = -1 if ta < 0 else 1
  ta_clipped = sign * min(abs(ta), max_charge_time)
  #=== Commit new charging time ===
  sol.commit_new_charge_time_after(pred, v, ta_clipped)
  # Compute energy bought diff
  old_energy_bought_after = sol.get_energy_bought_after(pred, v)
  new_energy_bought_after = min(problem_instance.soc_max - energy_lvl, time_to_energy(ta_clipped, problem_instance))
  db_left = new_energy_bought_after - old_energy_bought_after
  #=== Commit new energy bought ===
  sol.commit_new_energy_bought_after(pred, v, new_energy_bought_after)
  sol.electricity_cost += db_left * problem_instance.pe
  # Violation diff
  dv_left = compute_diff_viol_after(sol,
                                    pred,
                                    v,
                                    energy_lvl + new_energy_bought_after,
                                    problem_instance.min_soc,
                                    energy_dh_r)
  energy_lvl += new_energy_bought_after
  old_viol_after = sol.get_violation_after(pred, v)
  sol.commit_new_violation_after(pred, v, old_viol_after + dv_left)
  sol.violation_tot += dv_left
  sol.cost += dv_left * penalty + db_left * problem_instance.pe
  # Energy after deadhead : before r
  new_energy = energy_lvl - energy_dh_r
  #=== Commit new energy before inserted request r ===
  sol.commit_new_energy_before(r, v, new_energy)
  return new_energy

@njit(inline="always")
def commit_right_connection(energy_lvl: float, succ: int, r: int, v: int, sol: object, problem_instance: object, penalty: float = 0.):
  """ Computes and COMMITS differential in energy and violation for a new right connection.

      When inserting a request two connections are made : left and right.

      .. -> predecessor -> successor -> .., r would be inserted as
      .. -> predecessor -> r -> successor -> ..

      The right connection is r -> successor.

      Args:
          energy_lvl: incoming energy level
          r: request id to be inserted
          succ: request id of successor
          v: aircraft id
          sol: solution object
          problem_instance: problem object

      Returns:
          db_right: float, differential in right energy bought
          dv_right: float, differential in right violation
          new_energy: float, new energy at r
  """
  viol_before = compute_violation(energy_lvl, problem_instance.min_soc)  # for serving r
  sol.commit_new_violation_before(r, v, viol_before)
  #=== Commit violation before r ===
  energy_lvl -= problem_instance.get_service_energy(r)
  sol.commit_new_energy_after(r, v, energy_lvl)
  energy_dh_succ = problem_instance.get_connection_energy(r, succ)
  # Violation when deadheading from r to succ
  viol_after = 0
  if energy_dh_succ > 0:
    viol_after = compute_violation(energy_lvl, problem_instance.min_soc)
  #=== Commit violation after r ===
  sol.commit_new_violation_after(r, v, viol_after)
  sol.violation_tot += viol_after + viol_before
  sol.cost += (viol_after + viol_before) * penalty
  energy_lvl -= energy_dh_succ
  # Clipping the value of charging time in case too large for insertion.
  tb = sol.get_charging_time_before(succ, v)
  max_charge_time = problem_instance.get_max_charge_time(r, succ)
  sign = -1 if tb < 0 else 1
  tb_clipped = sign * min(abs(tb), max_charge_time)
  #=== Commit charging time before succ ===
  sol.commit_new_charge_time_before(succ, v, tb_clipped)
  # Differential in energy bought
  old_energy_bought_before = sol.get_energy_bought_before(succ, v)
  new_energy_bought_before = min(problem_instance.soc_max - energy_lvl, time_to_energy(tb_clipped, problem_instance))
  db_right = new_energy_bought_before - old_energy_bought_before
  sol.electricity_cost += db_right * problem_instance.pe
  sol.cost += db_right * problem_instance.pe
  #=== Commit new energy bought before succ ===
  sol.commit_new_energy_bought_before(succ, v, new_energy_bought_before)
  # Setting new energy value
  energy_lvl += new_energy_bought_before
  return energy_lvl

@njit(inline="always")
def evaluate_insertion(r: int,
                       v: int,
                       sol: object,
                       problem_instance: object,
                       find_insertion_slot=find_insertion_slot,
                       left_connection=left_connection,
                       right_connection=right_connection,
                       propagate_new_energy=propagate_new_energy):
  """ Evaluate the differential in cost of inserting r in v.

      Inserting request r in v, results in a differential in routing cost first,
      and a differential in energy cost / violation after. This function computes
      both.
      The charging times after the predecessor and before the successor are clipped
      when evaluating of commiting and insertion, otherwise we could have infeasible
      charging times. The charging times around r are 0 since it is not in the route.

      Args:
            r: request id to be inserted
            v: aircraft id
            sol: solution object
            problem_instance: problem object

      Returns:
            diff_routing: float, differential in routing cost
            diff_bought: float, differential in energy bought
            diff_viol: float, differential in violation
  """
  # print("Evaluating insertion")
  #-- Routing diff
  pred, succ, feas = find_insertion_slot(r, v, sol, problem_instance)
  # print(feas)
  if not (feas):
    return np.inf, 0., 0.
  #=== routing change ===
  if succ >= 0:
    diff_routing = problem_instance.costs[pred, r] + problem_instance.costs[r, succ] - problem_instance.costs[pred, succ]
  else:
    diff_routing = problem_instance.costs[pred, r]
  #-- Computing new energy at succ
  diff_bought = 0.
  diff_viol = 0.
  # print("Left connection")
  #-- Left connection : Connecting pred to r
  db_left, dv_left, energy_lvl = left_connection(pred,
                                                  r,
                                                  v,
                                                  sol,
                                                  problem_instance)
  diff_bought += db_left
  diff_viol += dv_left
  #-- Right connection : Serving r and connecting to succ
  if succ >= 0:
    db_right, dv_right, energy_lvl = right_connection(energy_lvl,
                                                      succ,
                                                      r,
                                                      v,
                                                      sol,
                                                      problem_instance)
    diff_viol += dv_right
    diff_bought += db_right
    #-- Propagating changes from succ with new energy
    db, dv = propagate_new_energy(v,
                                  succ,
                                  sol,
                                  energy_lvl,
                                  problem_instance)
    return diff_routing, diff_bought + db, diff_viol + dv

  #-- if there is no succ, finish by serving r
  diff_viol += compute_diff_viol_before(sol, r, v, energy_lvl, problem_instance.min_soc)
  db, new_bought_before = compute_diff_bought_before(sol, r, v, energy_lvl, problem_instance)
  diff_bought += db
  return diff_routing, diff_bought, diff_viol

@njit(inline="always")
def commit_insertion(r: int, v: int, sol: object, problem_instance: object, penalty: float = 0., commit_propagation_new_energy=commit_propagation_new_energy):
  """ COMMITS insertion of r in route of v.

      Inserting request r in v, results in a differential in routing cost first,
      and a differential in energy cost / violation after. This function computes
      both.
      The charging times after the predecessor and before the successor are clipped
      when evaluating of commiting and insertion, otherwise we could have infeasible
      charging times. The charging times around r are 0 since it is not in the route.

      Args:
            r: request id to be inserted
            v: aircraft id
            sol: solution object
            problem_instance: problem object

      Returns:
            diff_routing: float, differential in routing cost
            diff_bought: float, differential in energy bought
            diff_viol: float, differential in violation
  """
  #-- Routing diff
  # print("Start cost ", sol.cost, "Inserting", r, " in ", v)
  pred, succ, feas = find_insertion_slot(r, v, sol, problem_instance)
  if not (feas):
    raise ValueError("Attempt to commit an infeasible insertion !")
  diff_routing = problem_instance.costs[pred, r]
  #-- Comitting left connection
  e_lvl = commit_left_connection(pred, r, v, sol, problem_instance, penalty=penalty)
  if succ >= 0:
    #-- Comitting right connection
    e_lvl = commit_right_connection(e_lvl, succ, r, v, sol, problem_instance, penalty=penalty)
    #-- Comitting propagation
    commit_propagation_new_energy(v,
                                  succ,
                                  sol,
                                  e_lvl,
                                  problem_instance,
                                  penalty=penalty)
    sol.commit_new_pred(succ, v, r)
    sol.commit_new_succ(r, v, succ)
    diff_routing += problem_instance.costs[r, succ] - problem_instance.costs[pred, succ]
  elif succ < 0:
    # If there is no succ, need to account for r alone.
    old_viol_before = sol.get_violation_before(r, v)
    old_bought_before = sol.get_energy_bought_before(r, v)
    dv = compute_diff_viol_before(sol, r, v, e_lvl, problem_instance.min_soc)  # for serving r
    db, new_bought_before = compute_diff_bought_before(sol, r, v, e_lvl, problem_instance)
    #=== Commit before r ===
    sol.commit_new_energy_bought_before(r, v, new_bought_before)
    sol.commit_new_energy_before(r, v, e_lvl + new_bought_before)
    sol.commit_new_violation_before(r, v, old_viol_before + dv)
    sol.electricity_cost += db * problem_instance.pe
    sol.cost += db * problem_instance.pe + dv * penalty
    sol.violation_tot += dv

    e_lvl -= problem_instance.get_service_energy(r)
    sol.commit_new_energy_after(r, v, e_lvl)
  #-- Comitting routing change
  sol.commit_new_assignment(r, v)  # assigning r to v
  sol.commit_new_succ(pred, v, r)
  sol.commit_new_pred(r, v, pred)
  sol.routing_cost += diff_routing
  sol.cost += diff_routing - problem_instance.lbda_u
  sol.unserved_count -= 1
  # print("End cost ", sol.cost)
  return None


@njit(inline="always")
def evaluate_removal(r: int,
                     v: int,
                     sol: object,
                     problem_instance: object,
                     compute_diff_viol_after=compute_diff_viol_after,
                     compute_diff_viol_before=compute_diff_viol_before,
                     propagate_new_energy=propagate_new_energy):
  """ Evaluate the differential in cost of removing r in route of v.

      Removing request r in v, results in a differential in routing cost first,
      and a differential in energy cost / violation after. This function computes
      both.
      The charging times after the predecessor and before the successor are clipped
      when evaluating of commiting and insertion, otherwise we could have infeasible
      charging times.

      Args:
            r: request id to be inserted
            v: aircraft id
            sol: solution object
            problem_instance: problem object

      Returns:
            diff_routing: float, differential in routing cost
            diff_bought: float, differential in energy bought
            diff_viol: float, differential in violation
  """
  pred = sol.get_pred(r, v)
  succ = sol.get_succ(r, v)
  # Differential in routing cost only
  if succ >= 0:
    diff_routing = problem_instance.costs[pred, succ] - problem_instance.costs[pred, r] - problem_instance.costs[r, succ]
  else:
    diff_routing = - problem_instance.costs[pred, r]
  # Differential in electricity cost and violation
  diff_electricity_cost = - sol.get_energy_bought_before(r, v) - sol.get_energy_bought_after(r, v)
  diff_violation = -sol.get_violation_before(r, v) - sol.get_violation_after(r, v)
  diff_fast_charge = 0
  if sol.get_charging_time_after(r, v) > 0:
    diff_fast_charge -= 1
  if sol.get_charging_time_before(r, v) > 0:
    diff_fast_charge -= 1
  # Evaluating energy change after pred
  energy_lvl = sol.get_energy_after(pred, v)
  energy_lvl += sol.get_energy_bought_after(pred, v)
  diff_bought = 0.
  db, dv = 0., 0.
  if succ >= 0:
    energy_dh = problem_instance.get_connection_energy(pred, succ)
    diff_violation += compute_diff_viol_after(sol,
                                        pred,
                                        v,
                                        energy_lvl,
                                        problem_instance.min_soc,
                                        energy_dh)
    energy_lvl -= energy_dh
    # Evaluating energy change before succ
    diff_bought, new_bought_before = compute_diff_bought_before(sol,
                                                                succ,
                                                                v,
                                                                energy_lvl,
                                                                problem_instance)

    energy_lvl += new_bought_before

    # Propagating from succ
    db, dv = propagate_new_energy(v,
                                  succ,
                                  sol,
                                  energy_lvl,
                                  problem_instance)
  else:
    #succ is end of route marker : -1
    # then violaton and energy bought after pred should not count anymore
    dv -= sol.get_violation_after(pred, v)
    db -= sol.get_energy_bought_after(pred, v)
    if sol.get_charging_time_after(pred, v) > 0:
      diff_fast_charge -= 1
  return diff_routing, diff_bought + db, diff_violation + dv, diff_fast_charge

@njit(inline="always")
def commit_removal(r: int, v: int, sol: object, problem_instance: object, penalty: float = 0., commit_propagation_new_energy=commit_propagation_new_energy):
  """ COMMITS the differential in cost of removing r in route of v.

      Removing request r in v, results in a differential in routing cost first,
      and a differential in energy cost / violation after. This function computes
      both.
      The charging times after the predecessor and before the successor are clipped
      when evaluating of commiting and insertion, otherwise we could have infeasible
      charging times.

      Args:
            r: request id to be inserted
            v: aircraft id
            sol: solution object
            problem_instance: problem object

      Returns:
            diff_routing: float, differential in routing cost
            diff_bought: float, differential in energy bought
            diff_viol: float, differential in violation
  """
  # print(f"Removing {r} from {v}")
  pred = sol.get_pred(r, v)
  succ = sol.get_succ(r, v)
  # print(f"Pred is {pred}, Succ is {succ}")
  # Updates cost and violation after removal
  diff_routing = -problem_instance.costs[pred, r]
  if succ >= 0:
    diff_routing += problem_instance.costs[pred, succ] - problem_instance.costs[r, succ]
  tb = sol.get_charging_time_before(r, v)
  ta = sol.get_charging_time_after(r, v)
  df = 0
  if ta > 0:
    df += -1
  if tb > 0:
    df += -1
  diff_electricity_cost = - (sol.get_energy_bought_before(r, v) + sol.get_energy_bought_after(r, v)) * problem_instance.pe
  diff_violation = -sol.get_violation_before(r, v) - sol.get_violation_after(r, v)
  sol.routing_cost += diff_routing
  sol.electricity_cost += diff_electricity_cost
  sol.fast_charge_count += df
  sol.cost += diff_electricity_cost + diff_routing + problem_instance.lbda_u + diff_violation * penalty + df * problem_instance.lbda_f
  sol.unserved_count += 1
  sol.violation_tot += diff_violation
  # Reset all attributes related to r
  clear_node(r, v, sol)
  # Evaluating energy change after pred
  energy_lvl = sol.get_energy_after(pred, v)
  energy_lvl += sol.get_energy_bought_after(pred, v)
  if succ >= 0:
    energy_dh = problem_instance.get_connection_energy(pred, succ)
    diff_viol = compute_diff_viol_after(sol,
                                        pred,
                                        v,
                                        energy_lvl,
                                        problem_instance.min_soc,
                                        energy_dh)

    sol.commit_new_violation_after(pred, v, sol.get_violation_after(pred, v) + diff_viol)
    energy_lvl -= energy_dh
    db, new_bb = compute_diff_bought_before(sol, succ, v, energy_lvl, problem_instance)
    sol.commit_new_energy_bought_before(succ, v, new_bb)
    sol.violation_tot += diff_viol
    sol.cost += diff_viol * penalty
    diff_violation += diff_viol
    # Evaluating energy change before succ
    diff_bought, new_bought_before = compute_diff_bought_before(sol,
                                                                succ,
                                                                v,
                                                                energy_lvl,
                                                                problem_instance)
    sol.electricity_cost += (diff_bought + db) * problem_instance.pe
    sol.cost += (diff_bought + db) * problem_instance.pe
    sol.commit_new_energy_bought_before(succ, v, new_bought_before)
    #-- commiting new pred/succ
    sol.commit_new_succ(pred, v, succ)
    sol.commit_new_pred(succ, v, pred)
    #-- cost change
    energy_lvl += new_bought_before
    # Propagating from succ

    commit_propagation_new_energy(v,
                                  succ,
                                  sol,
                                  energy_lvl,
                                  problem_instance,
                                  penalty=penalty)
  else:
    sol.commit_new_succ(pred, v, succ)
    ta = sol.get_charging_time_after(pred, v)
    diff_fast_charge = 0
    if ta > 0:
      diff_fast_charge = -1
    sol.fast_charge_count += diff_fast_charge
    sol.cost += diff_fast_charge * problem_instance.lbda_f
    dv = sol.get_violation_after(pred, v)
    db = sol.get_energy_bought_after(pred, v)
    sol.electricity_cost -= db * problem_instance.pe
    sol.cost -= db * problem_instance.pe + dv * penalty
    sol.violation_tot -= dv
    sol.commit_new_violation_after(pred, v, 0.)
    sol.commit_new_charge_time_after(pred, v, 0.)
    sol.commit_new_energy_bought_after(pred, v, 0.)
  return None

@njit(inline="always")
def evaluate_new_start_location(loc: int,
                                v: int,
                                sol: object,
                                problem_instance: object,
                                get_start_point=get_start_point,
                                feasible_connection=feasible_connection):
  """ Evaluate new starting point for aircraft v.

      Will be used in the rotate neighborhood.

      Args:
          loc: new starting location for v
          v: aircraft id
          sol: solution object
          problem_instance: problem

      Returns:
          diff_routing: float, routing differential
          diff_bought: float, energy bought differential
          diff_viol: float, violation differential
          diff_unserved_count: int, unserved count differential
  """
  ori_start = get_start_point(v, problem_instance)
  if loc == ori_start:
    # if the starting point does not actually change
    return 0., 0., 0., 0, 0
  diff_routing = 0
  diff_bought = 0
  diff_viol = 0
  diff_unserved_count = 0
  diff_fast_charge = 0
  # check if aircraft can directly start its assigned route with this starting point
  pred = loc
  succ = sol.get_succ(ori_start, v)
  if succ < 0:
    # route was empty
    return diff_routing, diff_bought, diff_viol, diff_unserved_count, diff_fast_charge
  #removing previous first connection
  diff_routing -= problem_instance.costs[problem_instance.n_requests + v, succ]
  #removing subsequent unfeasible connections
  feas = feasible_connection(loc, succ, problem_instance.time_compatible)
  while not (feas):
    diff_routing -= problem_instance.costs[pred, succ]
    diff_bought -= sol.get_energy_bought_before(succ, v) + sol.get_energy_bought_after(succ, v)
    diff_viol -= sol.get_violation_before(succ, v) + sol.get_violation_after(succ, v)
    diff_unserved_count += 1
    if sol.get_charging_time_before(succ, v) > 0:
      diff_fast_charge -= 1
    if sol.get_charging_time_after(succ, v) > 0:
      diff_fast_charge -= 1
    pred = succ
    succ = sol.get_succ(succ, v)
    feas = feasible_connection(loc, succ, problem_instance.time_compatible)
  #-- the route will start at succ
  #-- First connection being from loc to succ
  diff_routing += problem_instance.costs[loc, succ]
  # new energy after connection
  energy_lvl = problem_instance.soc_max - problem_instance.get_connection_energy(loc, succ)
  old_bought_before = sol.get_energy_bought_before(succ, v)
  new_bought_before = min(problem_instance.soc_max - energy_lvl, old_bought_before)
  energy_lvl += new_bought_before
  diff_bought += new_bought_before - old_bought_before
  # -- Now change is initialized at before succ and we can propagate from there
  db, dv = propagate_new_energy(v,
                                succ,
                                sol,
                                energy_lvl,
                                problem_instance)
  return diff_routing, diff_bought + db, diff_viol + dv, diff_unserved_count, diff_fast_charge



