####
# Module containing neighborhoods functions
# Each neighborhood has its associated commit and shake function.
# Commit commits a move while shake choose a random feasible point in the neighborhood.
#
# Rotate Path / Shift request / Swap requests / Search on charging times /
####
from numba import njit, jit
import logging
import numpy as np
import sys
sys.path.append("../")
from utils_electric import (
    find_insertion_slot,
    aggregate_diff,
    golden_section_search,
    get_start_point,
    rotate_attributes,
    repair_rotation,
    reset_costs
)

from routing_charging.search_utils import copy_solution

from routing_charging.evaluation import (
    evaluate_insertion,
    commit_insertion,
    evaluate_removal,
    commit_removal,
    evaluate_charging_move_after,
    evaluate_charging_move_before,
    commit_charging_move_before,
    commit_charging_move_after,
    evaluate_new_start_location,
    commit_propagation_new_energy,
    commit_greedy_cost,
    compute_greedy_cost
)

@njit(parallel=False)
def shift_neigh(sol: object,
                problem_instance: object,
                move: int,
                penalty: float = 0.,
                first_best: bool = False,
                shake: bool = False,
                evaluate_insertion=evaluate_insertion,
                aggregate_diff=aggregate_diff,
                evaluate_removal=evaluate_removal):
  """ Returns first best shift move to make with associated reward.

      Move Description: Choose a request r, remove it from the aircraft
                        serving it and re-insert it elsewhere.
                        Requests and aircraft are explored in order previously
                        randomized and the first improvement is returned.

      Args:
          problem_instance: instance of the problem
          sol: solution instance
          penalty: battery soc violation penalty

      Returns:
          r: int, request_id to be shifted
          new_v: int, aircraft to serve r
          old_v: int, aircraft that served r before
          reward: float, differential of the move returned :
                         is interpreted as reward for the bandit
                         choosing the neighbors to use.
  """
  # print("Entering shift neighborhood")
  u = np.random.random()
  best_diff = 0
  r_shift = -1
  backup = -1
  old_v = 0
  new_v = 0
  for r in problem_instance.requests_ids:
    #if the request is currently not served, shift is an insertion
    if r_shift >= 0 and first_best:
      #first best stop
      r_shift, new_v, old_v, best_diff, move

    if sol.get_assigned_aircraft(r) < 0:
      for v in problem_instance.aircraft_ids:
        diff_routing, diff_bought, diff_viol = evaluate_insertion(r, v, sol, problem_instance)
        cand_diff = aggregate_diff(diff_routing,
                                   diff_bought,
                                   diff_viol,
                                   problem_instance,
                                   is_insertion=True,
                                   penalty=penalty)
        move += 1
        if shake and cand_diff < np.inf:
          return r, v, -1, cand_diff, move
        if cand_diff < best_diff:
          best_diff = cand_diff
          r_shift = r
          old_v = -1
          new_v = v
    else:
      aircraft_serving_r = sol.get_assigned_aircraft(r)
      diff_routing, diff_bought, diff_viol, diff_fast_charge = evaluate_removal(r, aircraft_serving_r, sol, problem_instance)
      cand_diff_rm = aggregate_diff(diff_routing,
                                   diff_bought,
                                   diff_viol,
                                   problem_instance,
                                   is_removal=True,
                                   diff_fast_charge=diff_fast_charge,
                                   penalty=penalty)
      move += 1
      if shake:
        backup = r
      if shake and cand_diff_rm < np.inf and u < 0.5:
        return r, -1, aircraft_serving_r, cand_diff_rm, move

      if cand_diff_rm < best_diff:
          best_diff = cand_diff_rm
          r_shift = r
          old_v = aircraft_serving_r
          new_v = -1

      #re-insert request in another aircraft
      for v in problem_instance.aircraft_ids:
        if v == aircraft_serving_r:
          continue
        move += 1
        diff_routing, diff_bought, diff_viol = evaluate_insertion(r, v, sol, problem_instance)
        cand_diff_add = aggregate_diff(diff_routing,
                                   diff_bought,
                                   diff_viol,
                                   problem_instance,
                                   is_insertion=True,
                                   penalty=penalty)
        cand_diff_shift = cand_diff_add + cand_diff_rm

        if shake and cand_diff_shift < np.inf:
          return r, v, aircraft_serving_r, cand_diff_shift, move

        if cand_diff_shift < best_diff:
          best_diff = cand_diff_shift
          r_shift = r
          old_v = aircraft_serving_r
          new_v = v


  #-- shaking by removal if did not find a shaking before
  if shake:
    aircraft_serving_r = sol.get_assigned_aircraft(backup)
    diff_routing, diff_bought, diff_viol, diff_fast_charge = evaluate_removal(backup, aircraft_serving_r, sol, problem_instance)
    cand_diff_rm = aggregate_diff(diff_routing,
                                  diff_bought,
                                  diff_viol,
                                  problem_instance,
                                  is_removal=True,
                                  diff_fast_charge=diff_fast_charge,
                                  penalty=penalty)
    move += 1
    return backup, -1, aircraft_serving_r, cand_diff_rm, move
  #best improvement
  return r_shift, new_v, old_v, best_diff, move

@njit(inline="always")
def commit_shift(r: int,
                 new_v: int,
                 old_v: int,
                 sol: object,
                 problem_instance: object,
                 penalty: float = 0.,
                 commit_removal=commit_removal,
                 commit_insertion=commit_insertion):
  """ COMMITS a shift move.

      Moving request r from aircraft old_v to new_v.

      Args:
          r: request id
          new_v: new aircraft id, -1 if none
          old_v: aicraft currently servingr, -1 if none
          sol: solution object
          problem_instance: problem instance

      Returns:
          None

    Sol is modified IN PLACE when running this.
  """
  if new_v >= 0 and old_v >= 0:
    commit_removal(r, old_v, sol, problem_instance, penalty=penalty)
    commit_insertion(r, new_v, sol, problem_instance, penalty=penalty)
  if new_v < 0 and old_v >= 0:
    commit_removal(r, old_v, sol, problem_instance, penalty=penalty)
  if new_v >= 0 and old_v < 0:
    commit_insertion(r, new_v, sol, problem_instance, penalty=penalty)
  return None


@njit(parallel=False)
def charging_neigh(sol: object,
                    problem_instance: object,
                    move: int,
                    penalty: float = 0.,
                    first_best: bool = False,
                    shake: bool = False,
                    golden_section_search=golden_section_search):
  """ Returns first best charging (before a req) move to make with associated reward.

      Move Description: Choose a request r, running a tri-section search
                        on the charging time before and after the request
                        to find the one that minimizes the differential of the move.
                        The first improving move is returned (i.e. with diff < 0),
                        slow charge mode is explored first.

      Args:
          sol: solution instance
          problem_instance: instance of the problem
          penalty: battery soc violation penalty

      Returns:
          new_time: float, new charging time
          r: int, request_id around which the move happens
          v: int, aircraft id involved in the move
          reward: float, differential of the move returned :
                         is interpreted as reward for the bandit
                         choosing the neighbors to use.
          loc: 1 to indicate after 0 to indicate before.
  """
  u = np.random.random()
  best_diff = -1
  c_time = 0
  best_r = -1
  best_v = -1
  loc = False
  for r in problem_instance.requests_ids:
    if best_r >= 0 and first_best:
      # stop at first improve
      c_time, best_r, best_v, best_diff, loc, move

    v = sol.get_assigned_aircraft(r)
    # if the request is currently not served, continue
    if v < 0:
      continue
    # retrieve successor and predecessor of r
    predecessor = sol.get_pred(r, v)
    successor = sol.get_succ(r, v)
    # -- Determine which modes are allowed to
    # allow mode switch only after a flight
    # Being consistent with milp formulation.
    allow_before = True
    allow_after = True
    before_con = problem_instance.get_connection_time(predecessor, r)
    if before_con == 0:
      if abs(sol.get_charging_time_after(predecessor, v)) > 0.001:
        allow_before = False
      # elif sol.get_charging_time_after(predecessor, v) < -0.9:
      #   allow_fast_before = False
    # Defining the range for the over which to search
    min_tb = 0
    max_tb = max(0, problem_instance.get_max_charge_time(predecessor, r) - abs(sol.get_charging_time_after(predecessor, v)))
    max_tb_slow = min(max_tb, (problem_instance.soc_max - sol.get_energy_before(r, v) + sol.get_energy_bought_before(r,v)) / problem_instance.gamma_s )
    max_tb_fast = min(max_tb, (problem_instance.soc_max - sol.get_energy_before(r, v) + sol.get_energy_bought_before(r,v)) / problem_instance.gamma_f )
    w_bs = max_tb_slow - min_tb  #width tb slow
    w_bf = max_tb_fast - min_tb  #width tb fast

    if successor >= 0:
      min_ta = 0
      max_ta = max(0, problem_instance.get_max_charge_time(r, successor) - abs(sol.get_charging_time_before(successor, v)))
      max_ta_slow = min(max_ta, (problem_instance.soc_max - sol.get_energy_after(r, v)) / problem_instance.gamma_s )
      max_ta_fast = min(max_ta, (problem_instance.soc_max - sol.get_energy_after(r, v)) / problem_instance.gamma_f )
      w_as = max_ta_slow - min_ta  #width ta slow
      w_af = max_ta_fast - min_ta  #width ta fast
      #--
      after_con = problem_instance.get_connection_time(r, successor)
      if after_con == 0:
        if abs(sol.get_charging_time_before(successor, v)) > 0.001:
          allow_after = False
        # elif sol.get_charging_time_before(successor, v) < -0.9:
        #   allow_fast_after = False
    if allow_before:
      # == Searching first in slow charge mode - before r
      found_slow_tb, diff_slow_b, move = golden_section_search(evaluate_charging_move_before,
                                                              - max_tb_slow,
                                                              - min_tb,
                                                              sol,
                                                              r,
                                                              v,
                                                              problem_instance,
                                                              move,
                                                              penalty=penalty)

      if shake and u < 0.5:
        #stopping at first improvement
        return found_slow_tb, r, v, diff_slow_b, False, move

      if diff_slow_b < best_diff:
        best_diff = diff_slow_b
        c_time = found_slow_tb
        best_r = r
        best_v = v
        loc = False

    if successor >= 0 and allow_after:
      found_slow_ta, diff_slow_a, move = golden_section_search(evaluate_charging_move_after,
                                                        - max_ta_slow,
                                                        - min_ta,
                                                        sol,
                                                        r,
                                                        v,
                                                        problem_instance,
                                                        move,
                                                        penalty=penalty)

      if shake:
        return found_slow_ta, r, v, diff_slow_a, True, move

      if diff_slow_a < best_diff:
        best_diff = diff_slow_a
        c_time = found_slow_ta
        best_r = r
        best_v = v
        loc = True

    if allow_before:
      # === Searching second in fast charge mode
      found_fast_tb, diff_fast_b, move = golden_section_search(evaluate_charging_move_before,
                                                              min_tb,
                                                              max_tb_fast,
                                                              sol,
                                                              r,
                                                              v,
                                                              problem_instance,
                                                              move,
                                                              penalty=penalty)

      if diff_fast_b < best_diff:
          best_diff = diff_fast_b
          c_time = found_fast_tb
          best_r = r
          best_v = v
          loc = False

    if successor >= 0 and allow_after:
      #if there is a successor, search in slow mode after r
      found_fast_ta, diff_fast_a, move = golden_section_search(evaluate_charging_move_after,
                                                                min_ta,
                                                                max_ta_fast,
                                                                sol,
                                                                r,
                                                                v,
                                                                problem_instance,
                                                                move,
                                                                penalty=penalty)

      if diff_fast_a < best_diff:
        best_diff = diff_fast_a
        c_time = found_fast_ta
        best_r = r
        best_v = v
        loc = True

  if best_v < 0:
    #no improvement found
    return 0, -1, -1, 0, False, move
  #return best
  return c_time, best_r, best_v, best_diff, loc, move

@njit(inline="always")
def commit_charging(new_time: float,
                    r: int,
                    v: int,
                    after: bool,
                    sol: object,
                    problem_instance: object,
                    penalty: float = 0.,
                    commit_charging_move_after=commit_charging_move_after,
                    commit_charging_move_before=commit_charging_move_before):
  """ COMMITS a charging move, obtained after running neighborhood charging.

      The charging time after or before r in the route of v will be modified in
      sol after running this.

      Args:
          found_time: charging time to set
          r: request id
          v: aircraft id
          sol: solution instance
          problem_instance: problem object
          penalty: battery soc violation penalty

      Returns:
          None
  """
  if r < 0:
    # neighborhood did not find any improvement
    return None
  if after:
    commit_charging_move_after(sol, new_time, v, r, problem_instance, penalty=penalty)
  else:
    commit_charging_move_before(sol, new_time, v, r, problem_instance, penalty=penalty)

@njit(parallel=False, inline="always")
def rotate_neigh(sol: object,
                problem_instance: object,
                move: int,
                penalty: float = 0.,
                first_best: bool = False,
                shake: bool = False,
                evaluate_new_start_location=evaluate_new_start_location,
                aggregate_diff=aggregate_diff):
  """ Returns first best rotate move to make with associated reward.
      Since aircraft are homogenous, we can evaluate the rotation differential
      by evaluating the differential when switching starting locations

      Args:

      Returns:
              i: int, order of the rotation
              diff: float, differential
  """
  nv = problem_instance.n_aircraft
  best_diff = 0.
  best_order = 0
  for i in problem_instance.aircraft_ids:
    # rotating paths by order i, iterating over possibilities in random order
    if i == 0:
      # no change in location
      continue
    diff = 0.
    for v in range(nv):
      move += 1
      loc = get_start_point((v + i) % nv, problem_instance)
      diff_routing, diff_bought, diff_viol, diff_unserved_count, diff_fast_charge = evaluate_new_start_location(loc, v, sol, problem_instance)
      diff_v = aggregate_diff(diff_routing,
                              diff_bought,
                              diff_viol,
                              problem_instance,
                              diff_fast_charge,
                              penalty=penalty)
      diff += diff_v + diff_unserved_count * problem_instance.lbda_u
    #stop at first improvement
    if first_best and (diff < 0 or shake):
      return i, diff, move

    if diff < best_diff:
      best_diff = diff
      best_order = i

  return best_order, best_diff, move

@njit(inline="always")
def commit_rotation(order: int,
                    sol: object,
                    problem_instance: object,
                    penalty: float = 0.,
                    rotate_attributes=rotate_attributes,
                    repair_rotation=repair_rotation,
                    reset_costs=reset_costs):
  """ COMMITS new starting point for aircraft v.

      Will be used in the rotate neighborhood.
      Sol is modified INPLACE after running this.

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
  #== First rotate all attributes
  rotate_attributes(sol, order, problem_instance)
  #== Repair beginning of routes
  repair_rotation(sol, problem_instance)
  #== Reset cost
  reset_costs(sol)
  #== Recompute all propagated values and cost
  for v in range(problem_instance.n_aircraft):
    startv = problem_instance.n_requests + v
    commit_propagation_new_energy(v,
                                  startv,
                                  sol,
                                  problem_instance.soc_max,
                                  problem_instance,
                                  penalty=penalty)
  # commit greedy cost
  commit_greedy_cost(sol, problem_instance, penalty=penalty)
  return None

@njit(parallel=False)
def swap_neigh(sol: object,
               sol_staged: object,
               problem_instance: object,
               move: int,
               penalty: float = 0.,
               first_best: bool = False,
               shake: bool = False,
               evaluate_removal=evaluate_removal,
               evaluate_insertion=evaluate_insertion,
               aggregate_diff=aggregate_diff,
               commit_removal=commit_removal,
               copy_solution=copy_solution):
  """ Returns first improve swap move to make.

      Args:
          sol: soluton instance current
          sol_staged: solution instance used for staging
          problem_instance: problem
          penalty: battery soc violation penalty

      Returns:
          r1:
          r2:
          reward:
  """
  best_diff = 0.
  best_r1 = -1
  best_r2 = -1
  best_v1 = -1
  best_v2 = -1
  for r1 in problem_instance.requests_ids:
    for r2 in problem_instance.requests_ids:
      if best_r1 >= 0 and first_best:
        #stop at first best
        return best_r1, best_r2, best_v1, best_v2, best_diff, move
      # print(f"Trying to swap {r1} and {r2}..")
      if r1 <= r2:
        continue
      v1 = sol.get_assigned_aircraft(r1)
      v2 = sol.get_assigned_aircraft(r2)
      if v1 < 0 or v2 < 0 or v1 == v2:
        continue
      # -- Evaluating swap move
      # -- replace staged solution attributes by the ones of sol
      copy_solution(sol, sol_staged)
      # remove requests
      diff_routing_v1, diff_bought_v1, diff_viol_v1, diff_fast_charge_v1 = evaluate_removal(r1, v1, sol_staged, problem_instance)
      diff_routing_v2, diff_bought_v2, diff_viol_v2, diff_fast_charge_v2 = evaluate_removal(r2, v2, sol_staged, problem_instance)


      cand_diff_rm_v1 = aggregate_diff(diff_routing_v1,
                                   diff_bought_v1,
                                   diff_viol_v1,
                                   problem_instance,
                                   is_removal=True,
                                   diff_fast_charge=diff_fast_charge_v1,
                                   penalty=penalty)

      cand_diff_rm_v2 = aggregate_diff(diff_routing_v2,
                                   diff_bought_v2,
                                   diff_viol_v2,
                                   problem_instance,
                                   is_removal=True,
                                   diff_fast_charge=diff_fast_charge_v2,
                                   penalty=penalty)

      cand_diff_rm = cand_diff_rm_v1 + cand_diff_rm_v2

      #-- commit the removals into staged solution
      commit_removal(r1, v1, sol_staged, problem_instance, penalty=penalty)
      commit_removal(r2, v2, sol_staged, problem_instance, penalty=penalty)
      #--- Now evaluate the re-insertions
      diff_routing_v1, diff_bought_v1, diff_viol_v1 = evaluate_insertion(r2, v1, sol_staged, problem_instance)
      diff_routing_v2, diff_bought_v2, diff_viol_v2 = evaluate_insertion(r1, v2, sol_staged, problem_instance)

      cand_diff_add_v1 = aggregate_diff(diff_routing_v1,
                                   diff_bought_v1,
                                   diff_viol_v1,
                                   problem_instance,
                                   is_insertion=True,
                                   penalty=penalty)

      cand_diff_add_v2 = aggregate_diff(diff_routing_v2,
                                   diff_bought_v2,
                                   diff_viol_v2,
                                   problem_instance,
                                   is_insertion=True,
                                   penalty=penalty)

      cand_diff_add = cand_diff_add_v2 + cand_diff_add_v1

      cand_diff = cand_diff_rm + cand_diff_add
      move += 4

      if shake and cand_diff < np.inf:
        return r1, r2, v1, v2, cand_diff, move

      if cand_diff < best_diff:
        best_diff = cand_diff
        best_r1 = r1
        best_r2 = r2
        best_v1 = v1
        best_v2 = v2

  return best_r1, best_r2, best_v1, best_v2, best_diff, move


@njit(inline="always")
def commit_swap(sol: object,
                r1: int,
                r2: int,
                v1: int,
                v2: int,
                problem_instance: object,
                penalty: float = 0.,
                commit_removal=commit_removal,
                commit_insertion=commit_insertion):
  """ COMMITS SWAP

  """
  if r1 >= 0:
    # first remove requests
    commit_removal(r1, v1, sol, problem_instance, penalty=penalty)
    commit_removal(r2, v2, sol, problem_instance, penalty=penalty)
    # Then add them
    commit_insertion(r2, v1, sol, problem_instance, penalty=penalty)
    commit_insertion(r1, v2, sol, problem_instance, penalty=penalty)
  return None


@njit(inline="always")
def apply_neigh(neigh_idx: int,
                random_sol: object,
                current_sol: object,
                staging_sol: object,
                problem_instance: object,
                move: int,
                first_best: bool = False,
                penalty: float = 0.,
                copy_solution=copy_solution,
                shift_neigh=shift_neigh,
                commit_shift=commit_shift,
                charging_neigh=charging_neigh,
                commit_charging=commit_charging,
                rotate_neigh=rotate_neigh,
                commit_rotation=commit_rotation,
                swap_neigh=swap_neigh,
                commit_swap=commit_swap):
    """ Applies a neighbourhoods function :
            - First, a random point x0, within the neighborhood is selected (shaking).
            - Second, a first improvement search returns a point improving w.r.t x0.

        Current and best solution are updated using the results.

        Args :
            neigh_idx: neighborhood id to use
            fail_curr: failure count for neigh_idx
            current_sol: current solution
            best_sol: best solution
            staging_sol: staging solution
            problem_instance: instance of the problem
            penalty: battery soc violation penalty

        Returns :
              fail_curr: int, updated fail count
              stag: int, updated stagantion
              reward: reward obtained in this round

    """
    # print("Neigh", neigh_idx)
    reward = 0
    if neigh_idx == 0:
      # apply neigh 0
      r, new_v, old_v, diff, move =  shift_neigh(random_sol,
                                            problem_instance,
                                            move,
                                            first_best=first_best,
                                            shake=False,
                                            penalty=penalty)
      reward = abs(diff / random_sol.cost)
      if reward > 0:
        #commit
        commit_shift(r,
                      new_v,
                      old_v,
                      random_sol,
                      problem_instance,
                      penalty=penalty)
    elif neigh_idx == 1:
      # apply neigh 1
      new_time, r, v, diff, after, move = charging_neigh(random_sol,
                                                   problem_instance,
                                                   move,
                                                   first_best=first_best,
                                                   shake=False,
                                                   penalty=penalty)
      reward = abs(diff / random_sol.cost)

      if reward > 0:
        commit_charging(new_time,
                        r,
                        v,
                        after,
                        random_sol,
                        problem_instance,
                        penalty=penalty)
    elif neigh_idx == 3:
      # apply neigh 3
      order, diff, move = rotate_neigh(random_sol,
                                  problem_instance,
                                  move,
                                  first_best=first_best,
                                  shake=False,
                                  penalty=penalty)
      reward = abs(diff / random_sol.cost)

      if reward > 0:
        commit_rotation(order,
                        random_sol,
                        problem_instance,
                        penalty=penalty)
    elif neigh_idx == 2:
      # apply neigh 2
      r1, r2, v1, v2, diff, move = swap_neigh(random_sol,
                                            staging_sol,
                                            problem_instance,
                                            move,
                                            first_best=first_best,
                                            shake=False,
                                            penalty=penalty)
      reward = abs(diff / random_sol.cost)

      if reward > 0:
        commit_swap(random_sol,
                    r1,
                    r2,
                    v1,
                    v2,
                    problem_instance,
                    penalty=penalty)

    if random_sol.cost < current_sol.cost:
      # improvement of best
      copy_solution(random_sol, current_sol)
    return reward, move

@njit(inline="always")
def shake(neigh_idx: int,
          random_sol,
          staging_sol,
          problem_instance: object,
          move,
          penalty: float = 0.,
          shift_neigh=shift_neigh,
          commit_shift=commit_shift,
          swap_neigh=swap_neigh,
          commit_swap = commit_swap):
  """ Returns a random point in the neighborhood neigh_idx of current_sol.

  """

  if neigh_idx == 0:
      # apply neigh 0
      r, new_v, old_v, diff, move =  shift_neigh(random_sol,
                                            problem_instance,
                                            move,
                                            shake=True,
                                            penalty=penalty)

      commit_shift(r,
                  new_v,
                  old_v,
                  random_sol,
                  problem_instance,
                  penalty=penalty)

  elif neigh_idx == 2:
    # apply neigh 2
    r1, r2, v1, v2, diff, move = swap_neigh(random_sol,
                                          staging_sol,
                                          problem_instance,
                                          move,
                                          shake=True,
                                          penalty=penalty)

    commit_swap(random_sol,
                  r1,
                  r2,
                  v1,
                  v2,
                  problem_instance,
                  penalty=penalty)


  return move


