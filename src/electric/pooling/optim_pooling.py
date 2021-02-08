from dataclasses import dataclass
import numpy as np
import time
from numba import njit, objmode, int64, float64
from numba.experimental import jitclass
#-- Pooling globals
ID = 0
MEAN_ARRIVAL = 1
NPAX = 2
QUANT = 3
MAX_DEP = 4
CLASS = 5

spec = [
    ('n', int64),
    ('C', int64),
    ('demands', float64[:, :]),
    ('lbda_p', int64),
    ('beam_width', int64),
    ('parents', float64[:, :]),
    ('children', float64[:, :]),
    ('children_costs', float64[:]),
    ('parents_costs', float64[:]),
    ('parents_usage', float64[:, :]),
    ('children_usage', float64[:, :]),
    ('loads', float64[:]),
    ('parents_flights', float64[:,:]),
    ('children_flights', float64[:,:]),
    ('parents_waiting_time', float64[:]),
    ('children_waiting_time', float64[:]),
    ('children_total_cost', float64[:]),
    ('parents_total_cost', float64[:]),
    ('alpha_regular', float64),
    ('alpha_premium', float64)
]

@jitclass(spec)
class Beam():
  """ Beam container.

      Contains a beam of N partitions.
      parents and children contains only feasible nodes. When violation
      are present, node is not stored.

      parents: contains N partitions, shape (N, n)
                  n being the number of points to cluster.
                  Element (i, j) is equal to the bin in which
                  point j is placed for partition i.
      children: contains E partitions, shape (E, n)
                  n being the number of points to cluster.
                  Element (i, j) is equal to the bin in which
                  point j is placed for partition i.

      usage: monitors bins' usage. Shape (n,), element i is 1
             if bin i is used.
      loads: monitors bins' load. Shape (n, ) element i is the load
             of bin i.
      flights: monitors bins' departure time. Shape (n, ) element i
              is the departure time of group in bin i
      waiting_times: monitors users' waiting time. Shape (n, ) element i
                     is the waiting time of user i. -1 if not assigned.
      cost_usage: number of bins used in parents
      cost_waiting: waiting cost in parents

  """
  def __init__(self,
              n: int,
              C: int,
              demands: np.ndarray,
              lbda_p: float,
              beam_width: int,
              parents: np.ndarray,
              children: np.ndarray,
              children_costs: np.ndarray,
              parents_costs: np.ndarray,
              parents_usage: np.ndarray,
              children_usage: np.ndarray,
              loads: np.ndarray,
              parents_flights: np.ndarray,
              children_flights: np.ndarray,
              parents_waiting_time: np.ndarray,
              children_waiting_time: np.ndarray,
              children_total_cost: np.ndarray,
              parents_total_cost: np.ndarray,
              alpha_regular: float,
              alpha_premium: float):
    self.n = n
    self.C = C
    self.demands = demands
    self.lbda_p = lbda_p
    self.beam_width = beam_width
    self.parents = parents
    self.children = children
    self.children_costs = children_costs
    self.parents_costs = parents_costs
    self.parents_usage = parents_usage
    self.children_usage = children_usage
    self.loads = loads
    self.parents_flights = parents_flights
    self.children_flights = children_flights
    self.parents_waiting_time = parents_waiting_time
    self.children_waiting_time = children_waiting_time
    self.children_total_cost = children_total_cost
    self.parents_total_cost = parents_total_cost
    self.alpha_regular = alpha_regular
    self.alpha_premium = alpha_premium



@njit(parallel=False, inline="always")
def add_demand_in_bin(beam: object,
                      start_index: int,
                      j: int,
                      i: int,
                      k: int,
                      parent: np.ndarray,
                      reached_first_unused: bool):
  """ Adds demand k in bin j from parent i.
      Updates children usage costs (i.e. cost of using bins)
      beam is modified INPLACE.
  """
  beam.children[start_index + j,:] = parent
  beam.children[start_index + j, k] = j
  if beam.parents_usage[i, j] == 0:
    beam.children_costs[start_index + j] = beam.parents_costs[i] + beam.lbda_p
    beam.children_total_cost[start_index + j] = beam.parents_total_cost[i] + beam.lbda_p
    reached_first_unused = True
  else:
    beam.children_costs[start_index + j] = beam.parents_costs[i]
    beam.children_total_cost[start_index + j] = beam.parents_total_cost[i]
  beam.children_usage[start_index + j, :] = beam.parents_usage[i, :]
  beam.children_usage[start_index + j, j] = 1
  return reached_first_unused


@njit(parallel=False, inline="always")
def update_waiting_cost(beam: object,
                        i: int,
                        j: int,
                        k: int,
                        dep: float,
                        weight: float,
                        start_index: int,
                        parent: np.ndarray):
  """ Updates waiting cost after adding demand k in bin j from parent i.
      beam is modified INPLACE.

  """
  diff_wt = 0
  if beam.parents_flights[i, j] > dep:
    # only the new demand increases the waiting time
    diff_wt += weight * (beam.parents_flights[i, j] - beam.demands[k, MEAN_ARRIVAL])
    beam.children_waiting_time[start_index + j] = beam.parents_waiting_time[i] + diff_wt
    beam.children_flights[start_index + j,:] = beam.parents_flights[i,:]
  else:
    # need to compute differential for all demands present in group
    old_dep = beam.parents_flights[i, j]
    diff_wt += weight * (dep - beam.demands[k, MEAN_ARRIVAL])
    for p in range(k):
      if parent[p] == j:
        wg = beam.alpha_premium if beam.demands[p, CLASS] == 1 else beam.alpha_regular
        diff_wt += wg * (dep - old_dep)

    beam.children_waiting_time[start_index + j] = beam.parents_waiting_time[i] + diff_wt
    #-- fights
    beam.children_flights[start_index + j, :] = beam.parents_flights[i, :]
    beam.children_flights[start_index + j, j] = dep
  #--
  beam.children_total_cost[start_index + j] += diff_wt
  return None


@njit(parallel=False, inline="always")
def filter_conflicts(beam: object,
                     j: int,
                     k: int,
                     conflicts: np.ndarray,
                     parent: np.ndarray):
  """ Returns feasibility of node when adding k in bin j from parent parent"""
  feas = True
  pax_in_j = 0
  for p in range(k):
    if parent[p] == j and conflicts[p, k] == 1:
      #-- conflict
      feas = False
    if parent[p] == j:
      pax_in_j += beam.demands[p, NPAX]
  if pax_in_j > beam.C - beam.demands[k, NPAX]:
    feas = False
  return feas

@njit(parallel=False, inline="always")
def compute_children(beam, i: int, k: int, start_index: int, conflicts: np.ndarray):
  """ Computes and cache all children of partition i
      when inserting k. Gives the last index used in
      children cache. Children's cost are also computed
      and stored in children_costs. """
  parent = beam.parents[i,:]
  weight = beam.alpha_premium if beam.demands[k, CLASS] == 1 else beam.alpha_regular
  dep = beam.demands[k, QUANT]
  nb_child = 0
  reached_first_unused = False
  for j in range(k + 1):
    #-- Adding demand k in bin j. The level of the tree is also k
    feas = filter_conflicts(beam,
                            j,
                            k,
                            conflicts,
                            parent)
    if not (feas):
      #-- go to next assignement
      start_index -= 1
      continue
    nb_child += 1
    reached_first_unused = add_demand_in_bin(beam,
                                             start_index,
                                             j,
                                             i,
                                             k,
                                             parent,
                                             reached_first_unused)

    update_waiting_cost(beam,
                        i,
                        j,
                        k,
                        dep,
                        weight,
                        start_index,
                        parent)
    #-- Check for early stop
    if reached_first_unused:
      #-- first unused bin has been reached - no need to check further.
      break
  return nb_child, start_index + j + 1

@njit(parallel=False, inline="always")
def compute_all_children(beam: object, conflicts: np.ndarray, k: int):
  """ Computes all children of current set of parents """
  current_index = 0
  lvl_children = 0
  nb_parents = np.sum(beam.parents[:, 0] > -1)
  for i in range(beam.beam_width):
    if beam.parents[i, 0] > -1:
      n_child, new_index = compute_children(beam, i, k, current_index, conflicts)
      lvl_children += n_child
      current_index = new_index
    else:
      break
  return new_index


@njit(inline="always")
def commit_children(beam: Beam, index_max: int):
  """ Puts best beam_width children as current parents. """
  if index_max < beam.beam_width:
    beam.parents[:,:] = -1
    beam.parents[:index_max,:] = beam.children[:index_max,:]
    beam.parents_costs[:] = 0
    beam.parents_costs[:index_max] = beam.children_costs[:index_max]
    beam.parents_usage[:,:] = 0
    beam.parents_usage[:index_max,:] = beam.children_usage[:index_max,:]
    beam.parents_flights[:,:] = 0
    beam.parents_flights[:index_max,:] = beam.children_flights[:index_max, :]
    beam.parents_waiting_time[:] = 0
    beam.parents_waiting_time[:index_max] = beam.children_waiting_time[:index_max]
    beam.parents_total_cost[:] = 0
    beam.parents_total_cost[:index_max] = beam.children_total_cost[:index_max]
  else:
    #-- sorting to get best children N children
    idx = np.argsort(beam.children_total_cost[:index_max])[:beam.beam_width]
    beam.parents[:,:] = beam.children[idx,:]
    beam.parents_costs[:] = beam.children_costs[idx]
    beam.parents_usage[:,:] = beam.children_usage[idx,:]
    beam.parents_flights[:,:] = beam.children_flights[idx, :]
    beam.parents_waiting_time[:] = beam.children_waiting_time[idx]
    beam.parents_total_cost[:] = beam.children_total_cost[idx]

@njit(inline="always")
def clear_children(beam: Beam):
  """ Resets children values to init """
  beam.children[:,:] = -1
  beam.children_costs[:] = 0
  beam.children_usage[:,:] = 0
  beam.children_flights[:,:] = 0
  beam.children_waiting_time[:] = 0
  beam.children_total_cost[:] = 0


def compute_greedy_cost(beam: Beam, pooling_instance: object):
  """ Greedy cost for parents. """
  parents_cost = []
  parents_group_usage = []
  parents_waiting_time = []
  for i in range(beam.beam_width):
    groups = np.unique(beam.parents[i,:])
    n_used = len(np.unique(beam.parents[i,:]))
    parents_group_usage.append(n_used * beam.lbda_p)
    wt = 0
    for k in range(beam.n):
      pax = np.where(beam.parents[i,:] == k)[0]
      if len(pax) == 0:
        continue
      dep = np.max(beam.demands[pax, QUANT])
      for p in pax:
        wp = pooling_instance.alpha_regular if beam.demands[p, CLASS] == 0 else pooling_instance.alpha_premium
        wt += wp * (dep - beam.demands[p, MEAN_ARRIVAL])
    parents_waiting_time.append(wt)
    parents_cost.append(wt + n_used * beam.lbda_p)
  return parents_cost, parents_group_usage, parents_waiting_time





@njit(parallel=False)
def beam_search(beam: Beam, conflicts: np.ndarray):
  """ Grows the search tree using beam """
  for level in range(1, beam.n):
    if level == 2:
      #####
      # Starting counting time after first iteration is done
      # to remove numba compilation time from measurements.
      # The loss in precision is negligble, the first iteration
      # contains only two possible child, so the computing of all children
      # is very cheap.
      #####
      # print("Starting time")
      with objmode(time_start='f8'):
        time_start = time.time()
    index_max = compute_all_children(beam, conflicts, level)
    commit_children(beam, index_max)
    clear_children(beam)
  with objmode(time_search='f8'):
    time_search = time.time() - time_start
  return time_search


# @njit(parallel=False)
def beam_search_increment(beam: Beam, conflicts: np.ndarray, level: int):
  """ Increments tree search by one at level level the search tree using beam """
  time_start = time.time()
  index_max = compute_all_children(beam, conflicts, level)
  commit_children(beam, index_max)
  clear_children(beam)
  time_search = time.time() - time_start
  return time_search


