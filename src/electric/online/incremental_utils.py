import sys
sys.path.append("/Users/mehdi/ATCP/test-expe/src/electric/")
import numpy as np
import click
from pooling import optim_pooling, pooling_utils
from routing_charging import optim_electric, evaluation, search_utils
import time
import utils_electric

#-- Globals
ORIGIN = 1
DEST = 2
ORIGIN_TIME = 3
DEST_TIME = 4

ID = 0
MEAN_ARRIVAL = 1
NPAX = 2
QUANT = 3
MAX_DEP = 4
CLASS = 5


def compute_pairwise_costs(problem_instance):
  """ """
  eta = 34
  requests = problem_instance.requests
  costs = np.zeros((problem_instance.r + problem_instance.v, problem_instance.r + problem_instance.v))
  for i in range(problem_instance.r + problem_instance.v):
    for j in range(problem_instance.r + problem_instance.v):
      if i != j:
        drr = problem_instance.delta if requests[i, DEST] == requests[j, ORIGIN] else 2 * problem_instance.delta
        feas = requests[j, 3] - requests[i, 4] + problem_instance.time[int(requests[i, 2]), int(requests[j, 1])] - drr > 0
        if feas:
          if requests[i, 2] != requests[j, 1]:
            costs[i, j] = eta * (problem_instance.time[int(requests[i, 2]), int(requests[j, 1])] + problem_instance.time[int(requests[j, 1]), int(requests[j, 2])]) + problem_instance.skyports[int(requests[j, 1]), 1] + problem_instance.skyports[int(requests[j, 2]), 1] - requests[j, 5]
          else:
            costs[i, j] = eta * problem_instance.time[int(requests[j, 1]), int(requests[j, 2])] + problem_instance.skyports[int(requests[j, 2]), 1] - requests[j, 5]
        else:
          costs[i, j] = 0
  return costs

def delete_request(req_id: int,
                   previous_sol: object,
                   problem_instance: object):
  """
  """
  #-- Delete from solution
  assigned = previous_sol.get_assigned_aircraft(req_id)
  if assigned >= 0:
    evaluation.commit_removal(req_id, assigned, previous_sol, problem_instance)
  previous_sol.assignement = np.delete(previous_sol.assignement, req_id, axis=0)
  previous_sol.assignementMap = np.delete(previous_sol.assignementMap, req_id, axis=0)
  previous_sol.assignementMap[previous_sol.assignementMap >= req_id] -= 1
  previous_sol.charging_times = np.delete(previous_sol.charging_times, req_id, axis=0)
  previous_sol.energy_levels = np.delete(previous_sol.energy_levels, req_id, axis=0)
  previous_sol.energy_bought = np.delete(previous_sol.energy_bought, req_id, axis=0)
  previous_sol.violation = np.delete(previous_sol.violation, req_id, axis=0)
  #-- Delete from problem instance
  problem_instance.requests = np.delete(problem_instance.requests, req_id, axis=0)
  problem_instance.costs = np.delete(problem_instance.costs, req_id, axis=0)
  problem_instance.costs = np.delete(problem_instance.costs, req_id, axis=1)
  problem_instance.time_compatible = np.delete(problem_instance.time_compatible, req_id, axis=0)
  problem_instance.time_compatible = np.delete(problem_instance.time_compatible, req_id, axis=1)
  problem_instance.r -= 1
  problem_instance.requests_ids = np.arange(problem_instance.r - 1)
  problem_instance.n_requests -= 1
  problem_instance.requests[:, 0] = np.arange(problem_instance.requests.shape[0])
  #--
  for v in problem_instance.aircraft_ids:
    start_v = problem_instance.n_requests + v
    evaluation.commit_propagation_new_energy(v,
                                            start_v,
                                            previous_sol,
                                            problem_instance.soc_max,
                                            problem_instance)
  evaluation.commit_greedy_cost(previous_sol, problem_instance)
  #-- Get random and staging sol with initiate to detect bugs
  _, best_sol, staging_sol, random_sol = utils_electric.init_vns(problem_instance.s,
                                                                  problem_instance.v,
                                                                  problem_instance.r,
                                                                  problem_instance,
                                                                  evaluation.commit_greedy_cost,
                                                                  evaluation.commit_propagation_new_energy,
                                                                  optim_electric.SolutionSls)
  return previous_sol, best_sol, staging_sol, random_sol

def add_request(problem_instance: object,
                previous_sol: object,
                ori: int,
                dest: int,
                dep_time: int,
                value: float):
  """
  """
  old_n_r = problem_instance.r
  n_v = problem_instance.v
  new_n_r = old_n_r + 1
  #-- Get arrival time
  arr_time = dep_time + problem_instance.time[ori, dest]
  #-- Edit problem instance
  #-- Determine loc insertion properly
  loc_insertion = old_n_r
  if dep_time <= problem_instance.requests[0, ORIGIN_TIME]:
    loc_insertion = 0
  else:
    for i in range(problem_instance.r-1):
      if problem_instance.requests[i, ORIGIN_TIME] <= dep_time <= problem_instance.requests[i+1, ORIGIN_TIME]:
        loc_insertion = i
        break
  #--
  # print(f"Loc insertion {loc_insertion}")
  new_row = np.array([0, ori, dest, dep_time, arr_time, value])
  problem_instance.n_requests = new_n_r
  problem_instance.r = new_n_r
  problem_instance.requests = np.insert(problem_instance.requests, loc_insertion, new_row, axis=0)

  #--
  problem_instance.requests[:, 0] = np.arange(problem_instance.requests.shape[0])
  problem_instance.requests_ids = np.arange(problem_instance.r)
  #-- costs and feasibilities
  problem_instance.costs = compute_pairwise_costs(problem_instance)
  problem_instance.time_compatible = utils_electric.process_compatibility(problem_instance.requests,
                                                                          problem_instance.aircraft,
                                                                          problem_instance.time,
                                                                          problem_instance.delta)


  #-- Edit previous sol to get new one
  previous_sol.assignement = np.insert(previous_sol.assignement, loc_insertion, -1)
  previous_sol.assignementMap = np.insert(previous_sol.assignementMap, loc_insertion, np.zeros((2, problem_instance.v)) - 1, axis=0)
  previous_sol.assignementMap[previous_sol.assignementMap >= loc_insertion] += 1
  previous_sol.charging_times = np.insert(previous_sol.charging_times, loc_insertion, np.zeros((problem_instance.v, 2)), axis=0)
  previous_sol.energy_levels = np.insert(previous_sol.energy_levels, loc_insertion, np.zeros((problem_instance.v, 2)), axis=0)
  previous_sol.energy_bought = np.insert(previous_sol.energy_bought, loc_insertion, np.zeros((problem_instance.v, 2)), axis=0)
  previous_sol.violation = np.insert(previous_sol.violation, loc_insertion, np.zeros((problem_instance.v, 2)), axis=0)
  previous_sol.violation_tot = 0
  previous_sol.unserved_count = 0
  previous_sol.fast_charge_count = 0
  previous_sol.cost = 0
  previous_sol.routing_cost = 0
  previous_sol.electricity_cost = 0

  for v in problem_instance.aircraft_ids:
    start_v = problem_instance.n_requests + v
    evaluation.commit_propagation_new_energy(v,
                                            start_v,
                                            previous_sol,
                                            problem_instance.soc_max,
                                            problem_instance)

  evaluation.commit_greedy_cost(previous_sol, problem_instance)
  #-- Get random and staging sol with initiate to detect bugs
  _, best_sol, staging_sol, random_sol = utils_electric.init_vns(problem_instance.s,
                                                                  problem_instance.v,
                                                                  problem_instance.r,
                                                                  problem_instance,
                                                                  evaluation.commit_greedy_cost,
                                                                  evaluation.commit_propagation_new_energy,
                                                                  optim_electric.SolutionSls)
  return previous_sol, best_sol, staging_sol, random_sol




def get_wt_class(beam: object,
                 parent_id: int,
                 pooling_instance: object):
  """

  """
  wt_premium = 0
  wt_regular = 0
  n_regular = 0
  n_premium = 0
  for i in range(beam.n):
    #--- Get group of demand i, flight info, deduce E[WT]
    k = int(beam.parents[parent_id, i])
    fk = beam.parents_flights[parent_id, k]
    wt = fk - pooling_instance.demands[i, MEAN_ARRIVAL]
    if pooling_instance.demands[i, CLASS] == 1:
      wt_premium += wt
      n_premium += 1
    else:
      wt_regular += wt
      n_regular += 1

  mean_regular = 0 if n_regular == 0 else wt_regular / n_regular
  mean_premium = 0 if n_premium == 0 else wt_premium / n_premium
  return mean_premium, mean_regular


def get_requests_table(beam: object,
                        parent_id: int,
                        n_group: int,
                        start_id: int,
                        ori: int,
                        dest: int,
                        travel_time: int,
                        unit_value: float = 80.):
  """
  """
  table = np.zeros((n_group, 6))
  n_req = 0
  for g in range(beam.n):
    if beam.parents_usage[parent_id, g] == 0:
      continue
    #-- Get group members
    members = np.where(beam.parents[parent_id,:] == g)[0]
    fg = beam.parents_flights[parent_id, g]
    request_value = unit_value * len(members)
    table[n_req, 0] = start_id + n_req + 1
    table[n_req, 1] = ori
    table[n_req, 2] = dest
    table[n_req, 3] = fg
    table[n_req, 4] = fg + travel_time
    table[n_req, 5] = request_value
    n_req += 1
  return table



def get_requests_from_beam(beam: object,
                           ori: int,
                           dest: int,
                           start_id: int,
                           travel_time: int,
                           pooling_instance: object
                           ):
  """ Extract different requests arrangement from beam on leg ori,dest.

  """
  beam_width = beam.parents.shape[0]
  group_numbers = np.sum(beam.parents_usage, axis=1)
  request_tables = np.zeros((int(group_numbers[0]), 6, beam_width)) - 1
  wt_table = np.zeros((beam_width, 2)) - 1
  schedule_seen = {}
  k_max = 0
  current_possibility = 0
  # print(f"Group number {group_numbers[0]}.")
  for i in range(beam_width):
    if group_numbers[i] > group_numbers[0]:
      # k_max = i
      break
    #-- Get hash for schedule
    flights = beam.parents_flights[i,:]
    flights = flights[flights > 0]
    flights = np.sort(flights)
    hash_flights = flights.tostring()
    #-- Check if schedule is already seen
    if hash_flights in schedule_seen:
      continue
    schedule_seen[hash_flights] = 1
    #-- Otherwise add a request table associated to it
    table = get_requests_table(beam,
                                i,
                                int(group_numbers[0]),
                                start_id,
                                ori,
                                dest,
                                travel_time)
    request_tables[:, :, current_possibility] = table
    #-- Get avg WT for both classes for current table
    avg_wt_premium, avg_wt_regular = get_wt_class(beam,
                                                  i,
                                                  pooling_instance)
    wt_table[current_possibility, 0] = avg_wt_premium
    wt_table[current_possibility, 1] = avg_wt_regular
    #--
    current_possibility += 1


  #--
  # print(f"Got {current_possibility} possible request tables...")
  request_tables = request_tables[:,:,:current_possibility]
  wt_table = wt_table[:current_possibility, :]
  return request_tables, wt_table


def simulate_all_demands(n: int, legs: list, problem_instance: object, seed = 7):
  """
  """
  np.random.seed(8)
  modes_one = [90, 150, 240]
  modes_two = [400, 500, 600]
  modes = [(90, 180), (250, 350), (400, 500), (540, 640), (300, 600), (200, 450)]
  instances_pooling = []
  D = [np.floor(n / len(legs)) for k in range(len(legs))]
  D[-1] += n - np.sum(D)
  leg_mode = {}
  for i in range(len(legs)):
    leg_mode[legs[i]] = modes[i]

  requests = np.empty((0, 6))
  for i in range(len(legs)):
    mo, mt = leg_mode[legs[i]]
    pooling_instance = pooling_utils.get_pooling_instance(int(D[i]),
                                                          seed=i,
                                                          mode_one=mo,
                                                          mode_two=mt,
                                                          over_day=False)
    beam = pooling_utils.init_beam(pooling_instance, beam_width=1000)
    conflicts = pooling_utils.get_conflicts(pooling_instance)
    _ = optim_pooling.beam_search(beam, conflicts)
    request_tables, _ = get_requests_from_beam(beam,
                                              legs[i][0],
                                              legs[i][1],
                                              0,
                                              problem_instance.time[legs[i][0], legs[i][1]],
                                              pooling_instance
                                              )
    #append
    requests = np.concatenate((requests, request_tables[:,:,0]), axis=0)
    instances_pooling.append(pooling_instance)
  #-- Sort and Append fake requests
  requests = requests[requests[:, ORIGIN_TIME].argsort()]
  requests = np.concatenate((requests, problem_instance.requests[problem_instance.r:,:]), axis=0)
  #-- update id
  requests[:, 0] = np.arange(requests.shape[0])
  #-- Update problem instance
  problem_instance.requests = requests.astype(np.int64)
  problem_instance.r = len(requests) - problem_instance.v
  problem_instance.requests_ids = np.arange(problem_instance.r)
  problem_instance.n_requests = len(requests) - problem_instance.v
  problem_instance.costs = compute_pairwise_costs(problem_instance)
  problem_instance.time_compatible = utils_electric.process_compatibility(problem_instance.requests,
                                                                          problem_instance.aircraft,
                                                                          problem_instance.time,
                                                                          problem_instance.delta)
  problem_instance.lbda_f = utils_electric.compute_fastcharge_coef(problem_instance)
  problem_instance.lbda_u = utils_electric.compute_service_coef(problem_instance)
  return leg_mode, instances_pooling




def simulate_one_demand(mode_one, mode_two):
  """ simulate one demand charac"""
  u = np.random.random()
  if u < 0.5:
    mean_arr = np.random.normal(mode_one, 20)
  else:
    mean_arr = np.random.normal(mode_two, 20)
  dist_to_quantile = [5, 7, 3]
  npax = [1, 2, 3, 4]
  classes = [0, 1]
  prop_premium = 0.2
  #-- sim
  pax = np.random.choice(npax, p=np.array([0.75, 0.2, 0.025, 0.025]))
  cat = np.random.choice(classes, p=np.array([1 - prop_premium, prop_premium]))
  dist = np.random.choice(dist_to_quantile, p=np.array([0.5, 0.1, 0.4]))
  max_dep = np.random.choice([10, 15, 20])
  has_max_dep = np.random.choice(classes, p=np.array([0.8, 0.2]))
  if has_max_dep == 1:
    max_dep_time = mean_arr + max_dep
  else:
    max_dep_time = 710
  return mean_arr, pax, mean_arr+dist, max_dep_time, cat

def add_demand(level: int,
                pooling_instance: object,
                mean_arr: int,
                npax: int,
                quant: int,
                max_dep: int,
                status: int):
  """ Adds one demand to instance and beam structure """
  old_d = pooling_instance.demands.shape[0]
  new_demand_row = np.round(np.array([old_d, mean_arr, npax, quant, max_dep, status]).reshape((1, 6)))
  #-- adding new demands
  pooling_instance.demands = np.concatenate((pooling_instance.demands, new_demand_row), axis=0)
  #-- recomputing conflicts
  conflicts = pooling_utils.get_conflicts(pooling_instance)
  #-- Updating beam structure
  #... Not updating because beam search is already really fast for the sizes we need
  return conflicts



def get_new_requests(d: int,
                    legs: list,
                    instances_pooling: list,
                    leg_mode: dict,
                    problem_instance: object,
                    seed: int):
  """ """
  np.random.seed(seed)
  incr_level = d - 1
  id_leg = np.random.randint(0, len(legs)-1)
  incr_leg = legs[id_leg]
  # print(f"Simulating d+1: {incr_level} on leg {incr_leg}. With modes {leg_mode[incr_leg]}.")
  mean_arr, npax, quant, max_dep, status = simulate_one_demand(leg_mode[incr_leg][0], leg_mode[incr_leg][1])
  conflicts = add_demand(incr_level,
                          instances_pooling[id_leg],
                          mean_arr,
                          npax,
                          quant,
                          max_dep,
                          status)

  beam = pooling_utils.init_beam(instances_pooling[id_leg], beam_width=1000)
  time_bs = optim_pooling.beam_search(beam, conflicts)
  old_req_leg = problem_instance.requests[np.logical_and(problem_instance.requests[:, ORIGIN] == incr_leg[0], problem_instance.requests[:, DEST] == incr_leg[1]), :]

  new_requests, _ = get_requests_from_beam(beam,
                                          incr_leg[0],
                                          incr_leg[1],
                                          0,
                                          problem_instance.time[incr_leg[0], incr_leg[1]],
                                          instances_pooling[id_leg]
                                          )
  new_requests = new_requests[:,:, 0]

  return new_requests, old_req_leg, incr_leg



def process_anytime(anytime_cache: np.ndarray, pooling_time: float):
  """ """
  time_steps = [5, 10, 15, 20, 25, 30, np.inf]
  accept_status = np.zeros((1, len(time_steps)))
  for i in range(len(time_steps)):
    aux = anytime_cache[anytime_cache[:, 3] + pooling_time <= time_steps[i], :]
    if len(aux) > 0:
      status = np.max(aux[:, 0]) == 1
      accept_status[0, i] = status * 1
    else:
      accept_status[0, i] = 0
  return accept_status


def remove_and_add(problem_instance: object,
                  new_requests: np.ndarray,
                  incr_leg: tuple,
                  best_sol: object):
  """ """
  #-- Delete old leg
  old_req_leg = problem_instance.requests[np.logical_and(problem_instance.requests[:, ORIGIN] == incr_leg[0], problem_instance.requests[:, DEST] == incr_leg[1]),:]
  old_n_r = 0
  for k in range(old_req_leg.shape[0]):
    for i in range(problem_instance.r):
      if problem_instance.requests[i, ORIGIN] == incr_leg[0] and problem_instance.requests[i, DEST] == incr_leg[1]:
        _, _, _, _ = delete_request(i, best_sol, problem_instance)
        old_n_r += 1

  #-- Add new req on this leg
  for i in range(new_requests.shape[0]):
    # print(f"Adding {new_requests[i, ORIGIN_TIME]}")
    old_best_sol, new_best_sol, staging_sol, random_sol = add_request(problem_instance,
                                                          best_sol,
                                                          incr_leg[0],
                                                          incr_leg[1],
                                                          new_requests[i, ORIGIN_TIME],
                                                          new_requests[i, 5])
  _,_,_,sol_backup = utils_electric.init_vns(problem_instance.s,
                                            problem_instance.v,
                                            problem_instance.r,
                                            problem_instance,
                                            evaluation.commit_greedy_cost,
                                            evaluation.commit_propagation_new_energy,
                                            optim_electric.SolutionSls)
  search_utils.copy_solution(best_sol, sol_backup)
  return old_best_sol, new_best_sol, staging_sol, random_sol, sol_backup



def save_solution(sol: object, path: str):
  """ Save a routing solution object to path """
  pref = "src/logs_results/cache_online/"
  np.save(f"{pref}assigment_{path}", sol.assignement)
  np.save(f"{pref}assigmentMap_{path}", sol.assignementMap)
  np.save(f"{pref}charging_{path}", sol.charging_times)
  np.save(f"{pref}energylvl_{path}", sol.energy_levels)
  np.save(f"{pref}energybought_{path}", sol.energy_bought)
  np.save(f"{pref}violation_{path}", sol.violation)
  cache_attr = [sol.violation_tot,
                sol.unserved_count,
                sol.fast_charge_count,
                sol.cost,
                sol.routing_cost,
                sol.electricity_cost]

  return cache_attr


def load_solution(sol: object, path: str, cache: list):
  """ Loads a routing solution """
  pref = "src/logs_results/cache_online/"
  sol.assignement = np.load(f"{pref}assigment_{path}.npy")
  sol.assignementMap = np.load(f"{pref}assigmentMap_{path}.npy")
  sol.charging_times = np.load(f"{pref}charging_{path}.npy")
  sol.energy_levels = np.load(f"{pref}energylvl_{path}.npy")
  sol.energy_bought = np.load(f"{pref}energybought_{path}.npy")
  sol.violation = np.load(f"{pref}violation_{path}.npy")
  sol.violation_tot = cache[0]
  sol.unserved_count = cache[1]
  sol.fast_charge_count = cache[2]
  sol.cost = cache[3]
  sol.routing_cost = cache[4]
  sol.electricity_cost = cache[5]
  return None


def save_pb_instance(problem_instance: object, path: str):
  """ """
  pref = "src/logs_results/cache_online/"
  np.save(f"{pref}requests_{path}", problem_instance.requests)
  np.save(f"{pref}costs_{path}", problem_instance.costs)
  np.save(f"{pref}timecompat_{path}", problem_instance.time_compatible)
  np.save(f"{pref}requestsids_{path}", problem_instance.requests_ids)

  cache = [problem_instance.r,
          problem_instance.n_requests]
  return cache


def load_pb_instance(problem_instance: object, path: str, cache: list):
  """ """
  pref = "src/logs_results/cache_online/"
  problem_instance.requests = np.load(f"{pref}requests_{path}.npy")
  problem_instance.costs = np.load(f"{pref}costs_{path}.npy")
  problem_instance.time_compatible = np.load(f"{pref}timecompat_{path}.npy")
  problem_instance.requests_ids = np.load(f"{pref}requestsids_{path}.npy")
  problem_instance.r, problem_instance.n_requests = cache[0], cache[1]


def save_pooling_ins(pooling_instance: object, path: str):
  """ """
  pref = "src/logs_results/cache_online/"
  np.save(f"{pref}demands_{path}", pooling_instance.demands)


def load_pooling_ins(pooling_instance: object, path: str):
  """ """
  pref = "src/logs_results/cache_online/"
  pooling_instance.demands = np.load(f"{pref}demands_{path}.npy")


