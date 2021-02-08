from dataclasses import dataclass
import numpy as np
import re
from pooling import optim_pooling
from numba import njit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#-- Pooling globals
ID = 0
MEAN_ARRIVAL = 1
NPAX = 2
QUANT = 3
MAX_DEP = 4
CLASS = 5



@dataclass
class PoolingInstance:
  """ Container for pooling problem instance """
  demands: np.ndarray
  capacity: int
  max_wait_premium: float
  max_wait_regular: float
  lbda_p: float
  alpha_premium: float
  alpha_regular: float
  ID: int = 0
  MEAN_ARRIVAL: int = 1
  NPAX: int = 2
  QUANT: int = 3
  MAX_DEP: int = 4
  CLASS: int = 5

@dataclass
class PoolingSolution:
  demands: np.ndarray = None
  x: np.ndarray = None
  y: np.ndarray = None
  f: np.ndarray = None
  usage: int = None
  overall_waiting: float = None

def get_pooling_instance(n: int,
                         seed: int = 7,
                         ap: float = 2.,
                         ar: float = 1.,
                         mwp: int = 15,
                         mwr: int = 25,
                         prop_premium: float = 0.2,
                         mode_one: float = 90.,
                         mode_two: float = 600.,
                         over_day: bool = True):
  """ Gets a pooling instance of n demands.

      Args:
          n: number of demands in instance
          seed: random seed

      Returns:
          demands: np.ndarray, table containing demands one per row, id | mean arrival time | npax | quantile 1-delta | max departure time | class
  """
  ID = 0
  MEAN_ARRIVAL = 1
  NPAX = 2
  QUANT = 3
  MAX_DEP = 4
  CLASS = 5
  np.random.seed(seed)
  demands = np.zeros((n, 6))
  time_steps = list(range(720))
  # drawing from a bi-modal 2/3 of the points and 1/3 uniformly over the day.
  if over_day:
    morning_batch = int(n / 2)
    evening_batch = int(n / 3)
    rest = n - morning_batch - evening_batch
    morning_demands = np.random.normal(mode_one, 20, morning_batch) # mode one
    evening_demands = np.random.normal(mode_two, 20, evening_batch) # mode two
    rest_demands = np.random.choice(time_steps[:-40], rest)  # all day
  else:
    morning_batch = int(n / 2)
    evening_batch = n - morning_batch
    morning_demands = np.random.normal(mode_one, 20, morning_batch) # mode one
    evening_demands = np.random.normal(mode_two, 20, evening_batch)  # mode two

  dist_to_quantile = [5, 7, 3]
  npax = [1, 2, 3, 4]
  classes = [0, 1]
  for i in range(n):
    pax = np.random.choice(npax, p=np.array([0.75, 0.2, 0.025, 0.025]))
    cat = np.random.choice(classes, p=np.array([1 - prop_premium, prop_premium]))
    dist = np.random.choice(dist_to_quantile, p=np.array([0.5, 0.1, 0.4]))
    max_dep = np.random.choice([10, 15, 20])
    has_max_dep = np.random.choice(classes, p=np.array([0.8, 0.2]))

    if i < morning_batch:
      demands[i, ID] = i
      demands[i, MEAN_ARRIVAL] = morning_demands[i]
      demands[i, NPAX] = pax
      demands[i, QUANT] = morning_demands[i] + dist
      if has_max_dep == 1:
        demands[i, MAX_DEP] = morning_demands[i] + max_dep
      else:
        demands[i, MAX_DEP] = 710
      demands[i, CLASS] = cat
    elif morning_batch <= i < morning_batch + evening_batch:
      j = i - morning_batch
      demands[i, ID] = i
      demands[i, MEAN_ARRIVAL] = evening_demands[j]
      demands[i, NPAX] = pax
      demands[i, QUANT] = evening_demands[j] + dist
      if has_max_dep == 1:
        demands[i, MAX_DEP] = evening_demands[j] + max_dep
      else:
        demands[i, MAX_DEP] = 710
      demands[i, CLASS] = cat
    elif morning_batch + evening_batch <= i:
      j = i - morning_batch - evening_batch
      demands[i, ID] = i
      demands[i, MEAN_ARRIVAL] = rest_demands[j]
      demands[i, NPAX] = pax
      demands[i, QUANT] = rest_demands[j] + dist
      if has_max_dep == 1:
        demands[i, MAX_DEP] = rest_demands[j] + max_dep
      else:
        demands[i, MAX_DEP] = 710
      demands[i, CLASS] = cat

  instance = PoolingInstance(demands=np.round(demands),
                            capacity=4,
                            max_wait_premium=mwp,
                            max_wait_regular=mwr,
                            lbda_p=1000.,
                            alpha_premium=ap,
                            alpha_regular=ar)

  return instance

def get_conflicts(pooling_instance: object):
  """ Computes conflicts between demands in pooling.
      Attempt to make the milp formulation faster """
  ID = 0
  MEAN_ARRIVAL = 1
  NPAX = 2
  QUANT = 3
  MAX_DEP = 4
  CLASS = 5
  n = pooling_instance.demands.shape[0]
  conflicts = np.ones((n, n)) + 1
  for i in range(n):
    for j in range(n):
      max_quant = max(pooling_instance.demands[i, QUANT], pooling_instance.demands[j, QUANT])
      status_i = pooling_instance.max_wait_premium if pooling_instance.demands[i, CLASS] else pooling_instance.max_wait_regular
      status_j = pooling_instance.max_wait_premium if pooling_instance.demands[j, CLASS] else pooling_instance.max_wait_regular
      if max_quant - pooling_instance.demands[i, MEAN_ARRIVAL] > status_i:
        conflicts[i, j] = 1
      if max_quant - pooling_instance.demands[j, MEAN_ARRIVAL] > status_j:
        conflicts[i, j] = 1
      if pooling_instance.demands[i, MAX_DEP] < max_quant:
        conflicts[i, j] = 1
      if pooling_instance.demands[j, MAX_DEP] < max_quant:
        conflicts[i, j] = 1
  return conflicts


def generate_pooling_mzn(pooling_instance: object, conflicts: np.ndarray, tag: str = ""):
  """ Generate mzn model file for pooling problem """
  n = pooling_instance.demands.shape[0]
  info = ["demands"]
  FILENAME = f"src/logs_results/instance_pooling/Model_Pooling_{n}{tag}.mzn"
  attr = list(filter(lambda a: not a.startswith('__'), dir(pooling_instance)))
  sheet = open(FILENAME, 'w')
  to_write = []
  # name = f"conflicts_info_1d ="
  # to_write.append(f"{name} {list(conflicts.flatten())};\n")
  for a in attr:
    if a in info:
      name = f"{a}_info_1d ="
      to_write.append(f"{name} {list(getattr(pooling_instance, a).flatten())};\n")
      to_write.append(f"int: n_{a} = {getattr(pooling_instance, a).shape[0]};\n")
    elif isinstance(getattr(pooling_instance, a), int):
      if a not in ["n_aircraft", "n_requests"]:
        name = f"int: {a} ="
        to_write.append(f"{name} {getattr(pooling_instance, a)};\n")
    elif isinstance(getattr(pooling_instance, a), float):
      name = f"float: {a} ="
      to_write.append(f"{name} {getattr(pooling_instance, a)};\n")
  sheet.writelines(to_write)
  with open("src/electric/model_constraints_pooling.mzn") as infile:
    sheet.write(infile.read())
  sheet.close()
  print("Pooling File written.")
  corresponding_output = f"src/logs_results/Gurobi_output_pool/output_{n}"
  return FILENAME, corresponding_output

def compute_load(sol: PoolingSolution, k: int, pooling_instance: PoolingInstance):
  """ Computes the loads, in term of pax of group k in sol."""
  pax = 0
  for i in range(pooling_instance.demands.shape[0]):
    if sol.x[i, k]:
      pax += pooling_instance.demands[i, pooling_instance.NPAX]
  if pax > pooling_instance.capacity:
    raise ValueError("Number of passengers is exceeding capacity. Check solution.")
  return pax

def compute_avg_load(sol: PoolingSolution, pooling_instance: PoolingInstance):
  """ Computes the average load of a solution sol"""
  loads = []
  for k in range(pooling_instance.demands.shape[0]):
    pax = compute_load(sol, k, pooling_instance)
    if pax > 0:
      loads.append(pax)
  # print(f"Loads : {loads}")
  return np.mean(loads), loads

def compute_wt(sol: PoolingSolution, pooling_instance: PoolingInstance):
  """ Computes the average load of a solution sol"""
  wt_premium = []
  wt_regular = []
  for i in range(pooling_instance.demands.shape[0]):
    for k in range(pooling_instance.demands.shape[0]):
      if sol.x[i, k]:
        wt = sol.f[k] - pooling_instance.demands[i, MEAN_ARRIVAL]
        if pooling_instance.demands[i, CLASS] == 1:
          wt_premium.append(wt)
        else:
          wt_regular.append(wt)
  return wt_premium, wt_regular

def init_beam(pooling_instance: object, beam_width: int = 20):
  """ """
  n = pooling_instance.demands.shape[0]
  w_init = pooling_instance.alpha_premium if pooling_instance.demands[0, CLASS] == 1 else pooling_instance.alpha_regular
  init_parents = np.ones((beam_width, n)) * -1
  init_parents[0, 0] = 0
  init_children = np.ones((n * beam_width, n)) * -1
  init_parents_costs = np.zeros(beam_width,)
  init_parents_costs[0] = pooling_instance.lbda_p
  init_children_costs = np.zeros((n * beam_width,))
  init_parents_usage = np.zeros((beam_width, n))
  init_parents_usage[0, 0] = 1
  init_children_usage = np.zeros((beam_width * n, n))
  init_loads = np.zeros((n,))
  init_parents_flights = np.zeros((beam_width, n))
  init_parents_flights[0, 0] = pooling_instance.demands[0, QUANT]
  init_children_flights = np.zeros((beam_width * n, n))
  init_parents_wt = np.zeros((beam_width,))
  init_parents_wt[0] = w_init * (pooling_instance.demands[0, QUANT] - pooling_instance.demands[0, MEAN_ARRIVAL])
  init_children_wt = np.zeros((beam_width * n,))
  init_total_cost_c = np.zeros((beam_width * n,))
  init_total_cost_p = np.zeros((beam_width,))
  init_total_cost_p[0] = pooling_instance.lbda_p + init_parents_wt[0] * w_init
  beam = optim_pooling.Beam(n,
                            pooling_instance.capacity,
                            pooling_instance.demands,
                            pooling_instance.lbda_p,
                            beam_width,
                            init_parents,
                            init_children,
                            init_children_costs,
                            init_parents_costs,
                            init_parents_usage,
                            init_children_usage,
                            init_loads,
                            init_parents_flights,
                            init_children_flights,
                            init_parents_wt,
                            init_children_wt,
                            init_total_cost_c,
                            init_total_cost_p,
                            pooling_instance.alpha_regular,
                            pooling_instance.alpha_premium
                            )
  return beam




@dataclass
class BeamStatistics:
  waiting_premium: list
  waiting_regular: list
  waiting_regular_with_premium: list
  waiting_regular_without_premium: list
  waiting_premium_without_regular: list
  waiting_premium_with_regular: list
  last_arrive_class: dict
  waiting_by_group: dict
  best_n_group: int
  n_best: int
  loads: np.ndarray
  frequency_regular_last_not_mixed: float
  frequency_regular_last_mixed: float

def compute_statistics(beam: object, pooling_instance: object):
  """ Get metrics on pooling solutions. """
  id_best = 0
  waiting_premium = []
  waiting_regular = []
  waiting_regular_with_premium = []
  waiting_regular_without_premium = []
  waiting_premium_without_regular = []
  waiting_premium_with_regular = []
  waiting_by_group = {0: [], 1: [], 2: [], 3: [], 4: []}
  mixity_groups = [0 for k in range(beam.n)] #gives mixity group for each flight
  group_with_premium = []
  group_with_regular = []
  group_alone = []
  loads = np.zeros((beam.n,))
  premium_demands = np.where(beam.demands[:, CLASS] == 1)[0]
  regular_demands = np.where(beam.demands[:, CLASS] == 0)[0]
  # print(f"Number of premium {len(premium_demands)}")
  #-- Get group id with premium
  for i in premium_demands:
    group_with_premium.append(beam.parents[id_best, i])
    #-- Get group with 0,1,2,3,4 premium demands in it
    mixity_groups[int(beam.parents[id_best, i])] += 1
  for i in regular_demands:
    group_with_regular.append(beam.parents[id_best, i])
  #-- Get grouped mixed to monitor which class arrives last in these groups
  mixed_group = []
  for g in group_with_premium:
    for i in regular_demands:
      if g == beam.parents[id_best, i]:
        mixed_group.append(g)
        break
  #--
  group_id, counts = np.unique(beam.parents[id_best, :], return_counts=True)
  count_map = {key: val for key, val in zip(group_id, counts)}
  #--
  last_arrive_quant = {g: -1 for g in mixed_group}
  last_arrive_class = {g: -1 for g in mixed_group}
  n_demand_only_regular = 0
  groups_only_regular = set(group_with_regular) - set(group_with_premium)
  n_regular_solo = 0
  for g in groups_only_regular:
    if count_map[g] == 1:
      n_regular_solo += 1

  n_group_only_regular = len(set(group_with_regular) - set(group_with_premium)) - n_regular_solo
  n_demand_regular_mixed = 0
  #-- Count number of groups
  group_numbers = np.sum(beam.parents_usage, axis=1)
  best_n_group = group_numbers[id_best]
  n_best = np.sum(group_numbers == best_n_group)
  #-- Retrieve all waiting times and loads
  for i in range(beam.n):
    loads[int(beam.parents[id_best, i])] += pooling_instance.demands[i, NPAX]
    #--- Get group of demand i, flight info, deduce E[WT]
    k = int(beam.parents[id_best, i])
    if count_map[k] < 2:
      #ignore demands that are not pooled to avoid biased statistics
      continue
    fk = beam.parents_flights[id_best, k]
    wt = fk - pooling_instance.demands[i, MEAN_ARRIVAL]
    #--- Update last arrival status
    if k in mixed_group:
      if pooling_instance.demands[i, CLASS] == 0:
        n_demand_regular_mixed += 1
      if pooling_instance.demands[i, QUANT] > last_arrive_quant[k]:
        last_arrive_quant[k] = pooling_instance.demands[i, QUANT]
        last_arrive_class[k] = pooling_instance.demands[i, CLASS]
    #---
    if pooling_instance.demands[i, CLASS] == 1:
      waiting_premium.append(wt)
      if k not in group_with_regular:
        waiting_premium_without_regular.append(wt)
      else:
        waiting_premium_with_regular.append(wt)
    else:
      waiting_regular.append(wt)
      if k in group_with_premium:
        waiting_regular_with_premium.append(wt)
      else:
        waiting_regular_without_premium.append(wt)
        n_demand_only_regular += 1
    #-- adding waiting time to mixity groups
    waiting_by_group[mixity_groups[k]].append(wt)

  #-- Frequencies of event regular last
  if n_demand_only_regular == 0:
    frequency_regular_last_not_mixed = -1
  else:
    frequency_regular_last_not_mixed = n_group_only_regular / n_demand_only_regular
  array_last = np.fromiter(last_arrive_class.values(), dtype=float)
  frequency_regular_last_mixed = np.sum(array_last < 1) / n_demand_regular_mixed

  # print(f"Frequence regular last in mixed group: {frequency_regular_last_mixed}. In not mixed group {frequency_regular_last_not_mixed}")
  stats = BeamStatistics(waiting_premium,
                        waiting_regular,
                        waiting_regular_with_premium,
                        waiting_regular_without_premium,
                        waiting_premium_without_regular,
                        waiting_premium_with_regular,
                        last_arrive_class,
                        waiting_by_group,
                        best_n_group,
                        n_best,
                        loads,
                        frequency_regular_last_not_mixed,
                        frequency_regular_last_mixed)
  # print(f"#Requests : {best_n_group} ({n_best} solutions) - WT regular {np.mean(waiting_regular)} - WT premium {np.mean(waiting_premium)} - WT regular with premium {np.mean(waiting_regular_with_premium)} - Avg Load {np.mean(loads[loads>0])}")
  return stats


def impact_alpha(ar_grid: list,
                 n: int,
                 beam_search: object,
                 mwr: int,
                 mwp: int,
                 prop_premium: float):
  """ """
  ap = 1.
  tag=f"PropP.{prop_premium}_mwr.{mwr}_mwp.{mwp}"
  WT_regular = {}
  WT_premium = {}
  for ar in ar_grid:
    WT_premium[ar] = []
    WT_regular[ar] = []
  for ar in ar_grid:
    print(f"=== Alpha regular {ar} ===")
    for seed in range(20):
      pooling_instance = get_pooling_instance(n,
                                              seed=seed,
                                              ap=ap,
                                              ar=ar,
                                              mwr=mwr,
                                              mwp=mwp,
                                              prop_premium=prop_premium)
      np.random.seed(seed)
      np.random.shuffle(pooling_instance.demands)
      beam = init_beam(pooling_instance, beam_width=1000)
      conflicts = get_conflicts(pooling_instance)
      time_bs = beam_search(beam, conflicts)
      stats = compute_statistics(beam, pooling_instance)

      WT_premium[ar] += stats.waiting_premium
      WT_regular[ar] += stats.waiting_regular

  print("Got all results")
  #-- Create figure
  plot_loc = f"src/logs_results/analysis/Impact_alpha_{n}_{tag}.png"
  plt.figure(figsize = (17, 9))
  means_premium = [np.mean(WT_premium[ar]) for ar in ar_grid]
  means_regular = [np.mean(WT_regular[ar]) for ar in ar_grid]
  err_regular = np.zeros((2, len(ar_grid)))
  err_premium = np.zeros((2, len(ar_grid)))
  max_overall = 0
  quant_up = 0.75
  quant_down = 0.25
  for i in range(len(ar_grid)):
    err_regular[0, i] = means_regular[i] - np.quantile(WT_regular[ar_grid[i]], quant_down)
    err_regular[1, i] = np.quantile(WT_regular[ar_grid[i]], quant_up) - means_regular[i]
    err_premium[0, i] = means_premium[i] - np.quantile(WT_premium[ar_grid[i]], quant_down)
    err_premium[1, i] = np.quantile(WT_premium[ar_grid[i]], quant_up) - means_premium[i]
    if max(WT_premium[ar_grid[i]]) > max_overall:
      max_overall = max(WT_premium[ar_grid[i]])
    if max(WT_regular[ar_grid[i]]) > max_overall:
      max_overall = max(WT_regular[ar_grid[i]])

  print(f"Max expected waiting time overall is {max_overall}.")
  plt.errorbar(ar_grid, means_regular, yerr = err_regular, linestyle = 'None', marker = 's', label="Regular", capsize=20)
  plt.errorbar(ar_grid, means_premium, yerr = err_premium, linestyle = 'None', marker = 'd', label="Premium", capsize=20)
  plt.legend(loc = "upper right")
  plt.xlabel('Penalty for WT Regular.')
  plt.ylabel('Expected Waiting times')
  plt.xlim(-.1, 1.1)
  plt.ylim(0, 25)
  plt.legend(title = 'Classes', prop={'size': 20})
  plt.grid(True)
  plt.savefig(plot_loc)
  print("figure saved.")



def impact_mixity(ar_grid: list,
                 n: int,
                 beam_search: object,
                 mwr: int,
                 mwp: int,
                 prop_premium: float):
  """ """
  ap = 1.
  tag=f"PropP.{prop_premium}_mwr.{mwr}_mwp.{mwp}"
  WT_regular_with_premium = {}
  WT_regular_without_premium = {}
  WT_premium_without_regular = {}
  WT_premium_with_regular = {}
  Frequencies_reg_mixed = {}
  Frequencies_reg_not_mixed = {}
  #--
  for ar in ar_grid:
    WT_regular_with_premium[ar] = []
    WT_regular_without_premium[ar] = []
    WT_premium_without_regular[ar] = []
    WT_premium_with_regular[ar] = []
    Frequencies_reg_mixed[ar] = []
    Frequencies_reg_not_mixed[ar] = []
  #--
  for ar in ar_grid:
    print(f"=== Alpha regular {ar} ===")
    for seed in range(20):
      pooling_instance = get_pooling_instance(n,
                                              seed=seed,
                                              ap=ap,
                                              ar=ar,
                                              mwr=mwr,
                                              mwp=mwp,
                                              prop_premium=prop_premium)
      np.random.seed(seed)
      np.random.shuffle(pooling_instance.demands)
      beam = init_beam(pooling_instance, beam_width=1000)
      conflicts = get_conflicts(pooling_instance)
      time_bs = beam_search(beam, conflicts)
      stats = compute_statistics(beam, pooling_instance)
      WT_regular_with_premium[ar] += stats.waiting_regular_with_premium
      WT_regular_without_premium[ar] += stats.waiting_regular_without_premium
      WT_premium_without_regular[ar] += stats.waiting_premium_without_regular
      WT_premium_with_regular[ar] += stats.waiting_premium_with_regular
      #-- Prop Premium in last arrival in mixed groups
      Frequencies_reg_mixed[ar].append(stats.frequency_regular_last_mixed)
      if stats.frequency_regular_last_not_mixed >= 0:
        Frequencies_reg_not_mixed[ar].append(stats.frequency_regular_last_not_mixed)

  print("Got all results")
  meanpointprops = dict(marker='D',
                      markeredgecolor='black',
                      markerfacecolor='red')
  #-- Create figure
  plot_loc = f"src/logs_results/analysis/Impact_mixity_{n}_{tag}.png"
  fig, ax = plt.subplots(figsize = (20, 9))
  groups = grouped_boxplots([[WT_regular_with_premium[ar], WT_regular_without_premium[ar], WT_premium_without_regular[ar], WT_premium_with_regular[ar]] for ar in ar_grid], ax=ax, patch_artist=True, meanprops=meanpointprops, showmeans=True, showfliers=False)
  colors = ['lavender', 'lightblue', 'bisque', 'lightgreen']
  for item in groups:
    for color, patch in zip(colors, item['boxes']):
      patch.set(facecolor=color)
  proxy_artists = groups[-1]['boxes']
  ax.legend(proxy_artists, ['Regular in mixed groups', 'Regular in Regular only groups', 'Premium in Premium only groups', 'Premium in mixed groups'], loc='upper right', prop={'size': 18})
  ax.set_xlabel('Alpha Regular', fontsize=20)
  ax.set_ylabel('Expected Waiting Time', fontsize=20)
  ax.set(axisbelow=True,
          xticklabels=['0.', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.'])
  ax.tick_params(axis='both', labelsize=20)

  # ax.grid(axis='y', ls='-', color='white', lw=2)
  fig.savefig(plot_loc)
  # -- Last arriving class figure
  plot_loc = f"src/logs_results/analysis/Last_Arrival_{n}_{tag}.png"
  fig, ax = plt.subplots(figsize = (20, 9))
  groups = grouped_boxplots([[Frequencies_reg_mixed[ar], Frequencies_reg_not_mixed[ar]] for ar in ar_grid], ax=ax, patch_artist=True, meanprops=meanpointprops, showmeans=True, showfliers=False)
  colors = ['lavender', 'lightblue']
  for item in groups:
    for color, patch in zip(colors, item['boxes']):
      patch.set(facecolor=color)
  proxy_artists = groups[-1]['boxes']
  ax.legend(proxy_artists, ['Regular in mixed groups', 'Regular in Regular only groups'], loc='upper right', prop={'size': 18})
  ax.set_xlabel('Alpha Regular', fontsize=20)
  ax.set_ylabel('Observed Probability for a Regular demand \n to be last of its group.', fontsize=20)
  ax.set(axisbelow=True,
          xticklabels=['0.', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.'])
  ax.tick_params(axis='both', labelsize=20)
  ax.set_ylim(0, 0.52)
  # ax.grid(axis='y', ls='-', color='white', lw=2)
  fig.savefig(plot_loc)



def impact_hard_constraints(mwp_grid: list,
                            ar_grid: list,
                            n: int,
                            beam_search: object,
                            mwr: int,
                            prop_premium: float):
  """ """
  ap = 1.
  tag = f"PropP.{prop_premium}_mwr.{mwr}_mwpgrid"
  grid_size = (len(mwp_grid), len(ar_grid))
  WT_regular = np.zeros(grid_size)
  WT_premium = np.zeros(grid_size)
  WT_regular_sup = np.zeros(grid_size)
  WT_premium_sup = np.zeros(grid_size)
  WT_regular_inf = np.zeros(grid_size)
  WT_premium_inf = np.zeros(grid_size)
  x_grid = np.zeros(grid_size)
  y_grid = np.zeros(grid_size)
  quant_sup = 0.75
  quant_inf = 0.25
  for i, mwp in enumerate(mwp_grid):
    for j, ar in enumerate(ar_grid):
      wt_reg = []
      wt_pre = []
      print(f"=== Alpha regular {ar}. MWP {mwp} ===")
      for seed in range(20):
        pooling_instance = get_pooling_instance(n,
                                                seed=seed,
                                                ap=ap,
                                                ar=ar,
                                                mwr=mwr,
                                                mwp=mwp,
                                                prop_premium=prop_premium)
        np.random.seed(seed)
        np.random.shuffle(pooling_instance.demands)
        beam = init_beam(pooling_instance, beam_width=1000)
        conflicts = get_conflicts(pooling_instance)
        time_bs = beam_search(beam, conflicts)
        stats = compute_statistics(beam, pooling_instance)

        wt_reg += stats.waiting_regular
        wt_pre += stats.waiting_premium

      # print(f"Mean regular: {np.mean(wt_reg)} - Mean premium {np.mean(wt_pre)}.")
      # print(f"Alpha regular is {ar}, max wt premium is {mwp}.")
      # print(f"Max : {max(stats.waiting_regular)}, {max(stats.waiting_premium)}")
      WT_regular[i, j] = np.mean(wt_reg)
      WT_premium[i, j] = np.mean(wt_pre)
      #---
      x_grid[i, j] = mwp
      y_grid[i, j] = ar
      # ---
      WT_regular_sup[i, j] = np.quantile(wt_reg, quant_sup) - np.mean(wt_reg)
      WT_regular_inf[i, j] = np.mean(wt_reg) - np.quantile(wt_reg, quant_inf)
      WT_premium_sup[i, j] = np.quantile(wt_pre, quant_sup) - np.mean(wt_pre)
      WT_premium_inf[i, j] = np.mean(wt_pre) - np.quantile(wt_pre, quant_inf)

  print("Got all results")
  # np.save(f"src/logs_results/analysis/Impact_hard_constraints_DATA_Regular_{n}", WT_regular)
  # np.save(f"src/logs_results/analysis/Impact_hard_constraints_DATA_Premium_{n}", WT_premium)
  # #-- Create figure
  # WT_regular = np.load(f"src/logs_results/analysis/Impact_hard_constraints_DATA_Regular_{n}.npy")
  # WT_premium = np.load(f"src/logs_results/analysis/Impact_hard_constraints_DATA_Premium_{n}.npy")

  #-- Premiums
  plot_loc = f"src/logs_results/analysis/Impact_hard_constraints_3D_Premium_{n}_{tag}.png"
  fig = plt.figure(figsize=(20, 12))
  mycmap = plt.get_cmap('plasma')
  ax = fig.add_subplot(111, projection='3d')
  surface = ax.plot_surface(x_grid, y_grid, WT_premium, cmap=mycmap)
  ax.view_init(45, 70)
  ax.set_xlabel("Maxmium Waiting Time for Premiums", fontsize=19, labelpad=25)
  ax.set_zlabel("Premium Expected Waiting time", fontsize=19, labelpad=25)
  ax.set_ylabel("Alpha Regular", fontsize=19, labelpad=25)
  cb = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
  cb.set_label(label='Premium Expected Waiting Time', fontsize=19)
  cb.ax.tick_params(labelsize=20)
  ax.tick_params(axis='both', labelsize=16, pad=16)

  plt.savefig(plot_loc)
  #-- Regulars
  plot_loc = f"src/logs_results/analysis/Impact_hard_constraints_3D_Regular_{n}_{tag}.png"
  fig = plt.figure(figsize=(20, 12))
  mycmap = plt.get_cmap('plasma')
  ax = fig.add_subplot(111, projection='3d')
  surface = ax.plot_surface(x_grid, y_grid, WT_regular, cmap=mycmap)
  ax.view_init(45, 70)
  ax.set_xlabel("Maxmium Waiting Time for Premiums", fontsize=19, labelpad=25)
  ax.set_zlabel("Regular Expected Waiting time", fontsize=19, labelpad=25)
  ax.set_ylabel("Alpha Regular", fontsize=19, labelpad=25)
  cb = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
  cb.set_label(label='Regular Expected Waiting Time', fontsize=19)
  cb.ax.tick_params(labelsize=20)
  ax.tick_params(axis='both', labelsize=16, pad=16)

  plt.savefig(plot_loc)
  print("figures saved.")


def grouped_boxplots(data_groups, ax=None, max_width=0.5, pad=0.05, **kwargs):
  if ax is None:
    ax = plt.gca()

  max_group_size = max(len(item) for item in data_groups)
  total_padding = pad * (max_group_size - 1)
  width = (max_width - total_padding) / max_group_size
  kwargs['widths'] = width

  def positions(group, i):
    span = width * len(group) + pad * (len(group) - 1)
    ends = (span - width) / 2
    x = np.linspace(-ends, ends, len(group))
    return x + i

  artists = []
  for i, group in enumerate(data_groups, start=1):
    artist = ax.boxplot(group, positions=positions(group, i), medianprops=dict(linewidth=0), **kwargs)
    artists.append(artist)

  ax.margins(0.05)
  ax.set(xticks=np.arange(len(data_groups)) + 1)
  ax.autoscale()
  return artists




