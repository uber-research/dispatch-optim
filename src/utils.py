
import numpy as np
import itertools
import operator
import matplotlib.pyplot as plt
import networkx as nx
from io import StringIO
import sys
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib import cm
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import numba
from numba import njit, jit
from numba.typed import List

def simulate_requests(
                      n: int,
                      fly_time,
                      locations,
                      T,
                      verbose: bool = True) :
  """ Simulates n flight requests giving origin, destination and time constraints for each one.
      The time constraint is implicitly defined via the authorized legs.

      Example : {("JFK", "Node 1", 3) : [(("JFK", 3), ("Node 1", 17))]} means that there is request to fly from
              JFK to Node 1 and there is only one authorized leg which is leaving from JFK at time 3 and arrives at Node 1
              at time 17. Third coordinate of the key give first possible starting time.
      --------------
      Args :
              n : int, number of request to simulate
              fly_time: dict, contains fly time between each pair of location in locations
              locations: list, contains id of all skyports
              T: list, contains time steps
              verbose: whether to print created requests while running.
      --------------
      Returns :
              R : dict, key is a tuple origin, destination, earliest_dep and value is a list of tuples giving authorized legs.
  """
  R = {}
  if verbose:
    print("Generating demands...")
    print("")
  for i in range(n):
    ori, dest = np.random.choice(locations, size=2, replace=False)  #select random ori, dest
    e_time = np.random.choice([k for k in range(int(len(T[5:-30]) / (n)) * (i), int(len(T[5:-30]) / n) * (i + 1))])
    if verbose:
      print(f"Demand {i} from {ori} to {dest} at {'{:02d}:{:02d}'.format(*divmod(e_time + 420, 60))}")
    #earliest time to begin service
    width = 5
    R[(ori, dest, e_time)] = []
    for w in range(width):
      R[(ori, dest, int(e_time))].append(((ori, int(e_time + w)), (dest, int(e_time + w + 14 + fly_time[(ori, dest)]))))

  return R

def create_arcs(
                nodes,
                r,
                fly_time: Dict[Tuple[str, str], int],
                refuel_times: Dict[str, float]):
  """ Creates the set of arcs, the waiting arcs, the service arcs and the deadhead arcs.
      An arc is a tuple (n, m) where n and m are two timed nodes.
      --------------
      Args :
              nodes : list, contains the set of nodes
              r : dict, contains the demands - expects format of simulate_requests()
              fly_time : dict, gives the fly time between locations
              refuel_times: dict, gives the times it takes to refuel at each skyport
      --------------
      Returns :
              A_w : list, contains the waiting arcs.
              A_s : list, contains the service arcs.
              A_f : list, contains the deadhead arcs.
              A_g : list, contains the refueling arcs

  """
  A_w, A_s, A_f, A_g = [], [], [], []
  for n in nodes:
    for m in nodes:
      if n[0] == m[0] and abs(n[1] - m[1]) == refuel_times[n[0]]:
        if n[1] < m[1]:
          A_g.append((n, m))
      if n[0] == m[0] and abs(n[1] - m[1]) == 1:
        if n[1] < m[1]:
          A_w.append((n, m))
      elif n[0] != m[0] and abs(n[1] - m[1]) == fly_time[(n[0], m[0])]:
        if n[1] < m[1]:
          A_f.append((n, m))
  for a in list(r.values()):
    A_s += a
  return A_w, A_s, A_f, A_g


def create_arcs_pruned(
                      nodes,
                      r,
                      fly_time,
                      refuel_times):
  """ Creates the set of arcs, the waiting arcs, the service arcs and the deadhead arcs.
      An arc is a tuple (n, m) where n and m are two timed nodes. Not all deaheads are considered,
      only a window around the requested legs.
      --------------
      Args :
              nodes : list, contains the set of nodes
              r : dict, contains the demands - expects format of simulate_requests()
              fly_time : dict, gives the fly time between locations
              refuel_time : dict, contains refuelling times for each locations
      --------------
      Returns :
              A_w : list, contains the waiting arcs.
              A_s : list, contains the service arcs.
              A_f : list, contains the deadhead arcs.
              A_g : list, contains the refueling arcs

  """
  A_w, A_s, A_f, A_g = [], [], [], []
  for n in nodes:
    for m in nodes:
      if n[0] == m[0] and abs(n[1] - m[1]) == refuel_times[n[0]]:
        if n[1] < m[1]:
          A_g.append((n, m))
      if n[0] == m[0] and abs(n[1] - m[1]) == 1:
        if n[1] < m[1]:
          A_w.append((n, m))
      elif n[0] != m[0] and abs(n[1] - m[1]) == fly_time[(n[0], m[0])]:
        #here we have a possible deadhead arcs we will add check wether it satisfies
        #one of the pruning conditions
        not_pruned = 0
        for req in r:
          for legs in r[req]:
            if legs[1][0] == n[0]:
              not_pruned = max((n[1] - legs[1][1] <= 15 + refuel_times[n[0]] + 5) * (n[1] - legs[1][1] >= 0) * 1, not_pruned)
            if legs[0][0] == m[0]:
              not_pruned = max((legs[0][1] - n[1] <= 15 + refuel_times[m[0]] + 5) * (legs[0][1] - m[1] >= 0) * 1, not_pruned)
        if n[1] < m[1] and not_pruned==1:
          A_f.append((n, m))
  for a in list(r.values()):
    A_s += a
  return A_w, A_s, A_f, A_g

@njit
def sort_ins(len_q, k_max, queue_swap, cand, cost_cand):
    """
        Implements a simple insertion sort algorithm. Insert a new element in queue_swap while
        maintaining it sorted and of size smaller than k_max.
        This function is used in the neighbourhoods functions in META_HEURISTIC in src/optim

        Args :
              len_q: int, lenght of queue_swap
              k_max: int,
              queue_swap: list, contains tuple
              cand: np.ndarray, candidate solution
              cost_cand: float, cost of candidate solution

        Returns:

              queue_swap: list, updated list with cand inserted if the lenght is less than k_max
              otherwise it is inserted only if there exist an element with greater cost already in
              queue_swap. Insertion sort.

    """
    if len_q < k_max:
        for id_l, l in enumerate(queue_swap):
            if id_l == len_q - 1 and l[0] < cost_cand:
                queue_swap.append((cost_cand, cand))
                break
            if id_l == len_q - 1 and l[0] >= cost_cand:
                queue_swap.insert(1, (cost_cand, cand))
                break

            if l[0] >= cost_cand:
                queue_swap.insert(id_l, (cost_cand, cand))
                break

    else:
        for id_l, l in enumerate(queue_swap):
            if l[0] >= cost_cand:
                queue_swap.insert(id_l, (cost_cand, cand))
                queue_swap = queue_swap[:-1]
                break
    return queue_swap


def create_cost(
                A_w,
                A_s,
                A_f,
                A_g,
                A_sink,
                carac_heli,
                beta: float,
                fee,
                helicopters,
                fly_time: Dict[Tuple[str, str], int],
                refuel_prices: Dict[str, float]):
  """ Assigns cost to each arc.
      --------------
      Args :
              A_w : list, contains the waiting arcs.
              A_s : list, contains the service arcs.
              A_f : list, contains the deadhead arcs.
              A_sink: list, arcs connecting the sinks
              carac_heli : dict, contains caracteristic of helicopters
              beta : float, gain from serving a demand
              fee : dict, landing fees
              helicopters : list, id of helicopters
              fly_time : dict, pairwise fly time between locations
              refuel_prices : dict, gives the cost in $ of refuelling at different locations

      --------------
      Returns :
              costs : dict, contains the cost associated with each arc

  """
  A = A_w + A_s + A_f + A_sink + A_g
  costs = {}
  for a in A:
    for h in helicopters:
      if a in A_w or a in A_sink:
        costs[(a, h)] = 0
      elif a in A_f:
        costs[(a, h)] = fee[a[1][0]] + carac_heli[h]["cost_per_min"] * fly_time[(a[0][0], a[1][0])]
      elif a in A_s:
        costs[(a, h)] = fee[a[1][0]] + carac_heli[h]["cost_per_min"] * fly_time[(a[0][0], a[1][0])] - beta
      elif a in A_g:
        costs[(a, h)] = refuel_prices[a[0][0]] #refueling price

  return costs


def connect_sink(
                sinks: Dict,
                locations,
                T):
  """ Connects ending nodes to artificial sink node
      --------------
      Args :
              sinks : dict, sink for each helicopter
              locations : list, contains locations
              T : list, time steps

      --------------
      Returns :
              sink_arcs : list, contains the arcs connecting the sink

  """
  sink_arcs = []
  for loc in locations:
    for s in sinks:
      sink_arcs.append(((loc, max(T)), (s[0], max(T))))
  return sink_arcs



def inbound(
            i: Tuple[str, int],
            arcs: Dict[str, Tuple[Tuple[str, int], Tuple[str, int]]],
            helicopters: List) -> List:
  """ Gives the list of arcs inbound to node i.
      --------------
      Args :
              i : tuple (location, time), represent a timed-node
              arcs : dict, contains all arcs
              helicopters : list of helicopters id
      --------------
      Returns :
              inb : list, contains all arcs inbound to node i

  """
  inb = []
  for h in helicopters:
    for a in arcs[h]:
      if a[1] == i:
        inb.append((a, h))
  return inb


def outbound(i, arcs, helicopters):
  """ Gives the list of arcs leaving node i.
      --------------
      Params :
              i : tuple (location, time), represent a timed-node
              arcs : dict, contains all arcs
              helicopters : list of helicopters
      --------------
      Returns :
              inb : list, contains all arcs leaving node i

  """
  outb = []
  for h in helicopters:
    for a in arcs[h]:
      if a[0] == i:
        outb.append((a, h))
  return outb


def get_helicopter_path(A, arcs, A_s, A_g, helicopters, carac_heli, T, v):
  """ Computes helicopters paths from solution. Prints a human-readable schedule
  for helicopters.
      --------------
      Params :
              A : list, contains all arcs
              arcs : pulp dict variable, contains arcs values after solving model
              A_s : list, contains all service arcs
              A_g : list, contains all refuelling arcs
              helicopters : list, contains all helicopters
              carac_heli : dict, caracteristics of helicopters
              T : list, time steps
              v : pulp dict variable, contains fuel levels

      --------------
      Returns :
              paths: dict, contains list of visited nodes for each helicopter.

  """
  paths = {h: [] for h in helicopters}
  print("====== Schedules =======")
  for h in helicopters:
    print("-------------")
    print("Helicopter ", h, " :")
    start = carac_heli[h]["start"]
    current_position = (start, 0)
    paths[h].append(current_position)
    if sum([arcs[h][a].varValue == 1 for a in A if a[0]==current_position]) == 0:
      print(h, " does not leave starting point. Not used in solution.")
      continue
    while current_position[1] < max(T):
      for a in A:
        if a[0] == current_position and arcs[h][a].varValue == 1:
          next_position = a[1]
          paths[h].append(next_position)
          if next_position[0] != current_position[0]:
            if a in A_s:
              print(h, " starts service in ", current_position[0], "at ", '{:02d}:{:02d}'.format(*divmod(870 + current_position[1], 60)), "and finishes in ", next_position[0], "at ", '{:02d}:{:02d}'.format(*divmod(870 + next_position[1], 60)), " - Fuel level at arrival: ", round(100 * v[h][next_position[0]][next_position[1]].varValue / carac_heli[h]["fuel_cap"], 2), "%")
            else:
              print(h, " leaves from ", current_position[0], "at ", '{:02d}:{:02d}'.format(*divmod(870 + current_position[1], 60)), "and arrives in ", next_position[0], "at ", '{:02d}:{:02d}'.format(*divmod(870 + next_position[1], 60)), " - Fuel level at arrival: ", round(100 * v[h][next_position[0]][next_position[1]].varValue / carac_heli[h]["fuel_cap"], 2), "%")
          if next_position[0] == current_position[0]:
            if a in A_g:
              print(h, " starts refueling in ", current_position[0], " at ",'{:02d}:{:02d}'.format(*divmod(870 + current_position[1], 60)), ", finishes at ", '{:02d}:{:02d}'.format(*divmod(870 + next_position[1], 60)), " - Fuel level at arrival: ", round(100 * v[h][next_position[0]][next_position[1]].varValue / carac_heli[h]["fuel_cap"], 2), "%")
          current_position = next_position
  return paths




def request_margins(paths, r, T, A_s, A_f, A_g, refuel_times, verbose=True):
  """ Computes margins for demand : how far in the future/past demands can be moved without breaking the current paths.
      --------------
      Params :
            paths : dict, contains path of helicopter. Output of get_helicopter_path()
            r : dict, contains requests/demands
            T : list, time steps
            A_s : list, contains all service arcs
            A_g : list, contains all refuelling arcs
            A_f : list, contains all deadhead arcs
            refuel_times : dict, contains refuelling time for locations
            verbose : bool, for printing log

      --------------
      Returns :
            r_margins : dict, contains margins expressed in minutes for each request/demand in r.

  """
  r_margins = {req: {"plus":0, "minus":0} for req in r}
  served = {h: [] for h in paths}
  deadheads = {h: [] for h in paths}
  refuel = {h:[] for h in paths}
  for h in paths:
    route = paths[h]
    if route == []:
      continue
    for i in range(1, len(route)):
      current_position = route[i - 1]
      next_position = route[i]
      a = (current_position, next_position)
      if a in A_s:
        req = [req for req in r if a in r[req]][0]
        served[h].append((req, a[1][1] - a[0][1]))
      if a in A_f:
        deadheads[h].append(a)
      if a in A_g:
        refuel[h].append(a)
  for h in served:
    if served[h] == []:
      continue
    first, time_first = served[h][0]
    last, time_last = served[h][len(served[h]) - 1]
    init_fuel = sum([refuel_times[a[0][0]] for a in refuel[h] if a[0][1] < time_first])
    init_dh = sum([a[1][1] - a[0][1] for a in deadheads[h] if a[0][1] < time_first])
    r_margins[first]["minus"] = max(time_first - init_fuel - init_dh, 0)
    r_margins[last]["plus"] = max(max(T) - (last[2] + time_last), 0)
    for i in range(1, len(served[h])):
      prev, time_prev = served[h][i-1]
      curr, _ = served[h][i]
      in_between_dh = [a for a in deadheads[h] if prev[2] <= a[0][1] and a[1][1] <= curr[2]]
      in_between_rf = [a for a in refuel[h] if prev[2] <= a[0][1] and a[1][1] <= curr[2]]
      rf_time = refuel_times[in_between_rf[0][0][0]] if in_between_rf != [] else 0
      if in_between_dh == []:
        r_margins[prev]["plus"] = max(curr[2] - (prev[2] + 5 + time_prev) - rf_time, 0)
        r_margins[curr]["minus"] = max(curr[2] - (prev[2] + 5 + time_prev) - rf_time, 0)
      else:
        dh_time = sum([a[1][1] - a[0][1] for a in in_between_dh])
        r_margins[prev]["plus"] = max(curr[2] - (prev[2] + 5 + time_prev) - rf_time - dh_time, 0)
        r_margins[curr]["minus"] = max(curr[2] - (prev[2] + 5 + time_prev) - rf_time - dh_time, 0)
  if verbose:
    print("")
    print("-----------")
    print("Margins : Number of minutes by which a request can be moved in time without breaking the current solution paths.")
    print("Request margins information :")
    for req in r_margins:
      print("Request from ", req[0], " to ", req[1], " at ", '{:02d}:{:02d}'.format(*divmod(420 + req[2], 60)), " can be postponed by up to", r_margins[req]["plus"], " minutes or start up to ", r_margins[req]["minus"], " minutes earlier.")
  return r_margins





def viz_graph(nodes, arcs, A, locations, helicopters, A_s, A_g, A_f, colors, r):
  """ Viz functions for the time expanded network, schedule is represented on graph.
      --------------
      Params :
            nodes : list, contains all nodes
            arcs : pulp dict variable, contains arcs value in solution
            A : list, contains all arcs
            locations : list, contains all locations
            helicopters : list, contains helicopters
            A_g : list, contains all refuelling arcs
            A_s : list, contains all service arcs
            colors : dict, colors to represent helicopters on graph
            r : dict, contains all requests/demands

      --------------
      Returns :
            fig : matplotlib figure to be plotted/saved

  """
  img=mpimg.imread('image/icon.png')
  G = nx.DiGraph()
  for n in nodes:
    G.add_node(n, pos=(n[1], locations.index(n[0])))
  G.add_edges_from(A)
  for h in helicopters:
    for e in A_g:
        if arcs[h][e].varValue == 1 :
            G.add_edge(e[0], e[1], image=img, size=0.09)



  pos = nx.get_node_attributes(G, 'pos')
  fig, ax = plt.subplots(figsize=(15, 7))
  plt.title("Graph Schedule, Rand." + str(len(r)) + ".H" + str(len(helicopters)))

  nx.draw_networkx_nodes(G, pos, node_size=5, ax=ax, node_color='black')
  for h in helicopters:
    nx.draw_networkx_edges(G,pos,
                          edgelist=[e for e in A if arcs[h][e].varValue == 1 and not(e in A_f)],
                          width=4, alpha=1, edge_color=colors[h])
    nx.draw_networkx_edges(G,pos,
                          edgelist=[e for e in A if arcs[h][e].varValue == 1 and e in A_f],
                          width=1, alpha=1, edge_color=colors[h], style="dotted")


  nx.draw_networkx_edges(G,pos,
                        edgelist=[e for e in A_s],
                        width=1, alpha=0.5, edge_color='g')


  ax.set_axis_on()
  ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
  plt.xlabel("Time")
  plt.yticks([i for i in range(len(locations))], locations)
  plt.xticks([i for i in range(210)][::15], ['{:02d}:{:02d}'.format(*divmod(870 + i, 60)) for i in range(210)][::15])


  ax2=plt.gca()
  fig2=plt.gcf()
  label_pos = 0.5 # middle of edge, halfway between nodes
  trans = ax2.transData.transform
  trans2 = fig2.transFigure.inverted().transform
  imsize = 0.1  # this is the image size
  rf = []
  for h in helicopters:
      rf += [e for e in A_g if arcs[h][e].varValue == 1]
  for (n1, n2) in rf:
      (x1,y1) = pos[n1]
      (x2,y2) = pos[n2]
      (x,y) = (x1 * label_pos + x2 * (1.0 - label_pos),
              y1 * label_pos + y2 * (1.0 - label_pos))
      xx,yy = trans((x,y)) # figure coordinates
      xa,ya = trans2((xx,yy)) # axes coordinates
      imsize = G[n1][n2]['size']
      img =  G[n1][n2]['image']
      a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
      a.imshow(img)
      a.set_aspect('equal')
      a.axis('off')



  return fig

def update_slot(paths, slots, helicopters, T, A_s, A_f, A_g, A_w, v, fly_time, refuel_times, carac_heli, min_fuel):
  """ Updates the status of different slots in the booking table.
      --------------
      Params :
            paths : dict, contains path of helicopter. Output of get_helicopter_path()
            slots : dict, contains previous slots status
            helicopters : list, contains all helicopters
            T : list, time steps
            A_s : list, contains all service arcs
            A_g : list, contains all refuelling arcs
            A_f : list, contains all deadhead arcs
            A_w : list, contains all waiting arcs
            v : pulp dict variable, contains fuel levels
            fly_time : dict, fly time between locations
            refuel_times : dict, contains refuelling time for locations
            carac_heli : dict, contains helicopters caracteristics
            min_fuel : float, minimum quantity of fuel to takeoff

      --------------
      Returns :
            slots : dict, updated version

  """
  in_service = {h: [] for h in helicopters}
  deadheading = {leg: [] for leg in slots}
  for h in in_service:
    route = paths[h]
    for i in range(1, len(route)):
      if (route[i-1], route[i]) in A_s:
        in_service[h] += [k for k in range(route[i - 1][1], route[i][1] + 1)]
      elif (route[i - 1], route[i]) in A_f:
        slack_right = [k for k in itertools.takewhile(lambda x: route[i-x][0] == route[i-1][0], range(1, len(route[:i-1])))]
        slack_left = [k for k in itertools.takewhile(lambda x:route[x + i][0] == route[i][0], range(len(route[i:])))]
        if len(slack_left) + len(slack_right) > 25:
          deadheading[(route[i-1][0], route[i][0])].append(route[i-1][1])
  serving = set(T)
  for h in in_service:
    serving = serving.intersection(set(in_service[h]))
  for leg in slots:
    for s in slots[leg]:
      ins_test = insertion_test(paths, T, A_s, A_g, A_w, (leg, s), v, fly_time, refuel_times, carac_heli, min_fuel)

      if ins_test and slots[leg][s] != "booked" and slots[leg][s] != "not feasible":
        slots[leg][s] = "bookable"
      if not (ins_test) and slots[leg][s] != "booked":
        slots[leg][s] = "not feasible"
      if np.sum([deadheading[leg][i] in range(s, s + 5) for i in range(len(deadheading[leg]))]) > 0:
        #check deadhead slack
        slots[leg][s] = "revenu oppotunity"

  return slots



def insertion_test(paths, T, A_s, A_g, A_w, new_slot, v, fly_time, refuel_times, carac_heli, min_fuel):
  """ Test for the feasibility of inserting a new request in a current schedule, accounting for refuelling.
      --------------
      Params :
            paths : dict, contains path of helicopter. Output of get_helicopter_path()
            T : list, time steps
            A_s : list, contains all service arcs
            A_g : list, contains all refuelling arcs
            A_w : list, contains all waiting arcs
            new_slot : tuple, contains slot to be tested (leg, time)
            v : pulp dict variable, contains fuel levels
            fly_time : dict, fly time between locations
            refuel_times : dict, contains refuelling time for locations
            carac_heli : dict, contains helicopters caracteristics
            min_fuel : float, minimum quantity of fuel to takeoff

      --------------
      Returns :
            bool : True iif insertion in feasible.
  """

  feasibility = {h: True for h in paths}
  leg, e_time = new_slot
  left_pos = {h: (carac_heli[h]["start"], 0) for h in paths}
  right_pos = {h: paths[h][-1] for h in paths}
  next_refuel = {h: paths[h][-1] for h in paths}
  in_service = {h: [] for h in paths}
  in_refuel = {h: [] for h in paths}
  if e_time + fly_time[leg] + 14 + 5 > 209:
    #out of service range - This is an assumption for now - probably more tricky to check.
    return False
  for h in paths:
    if len(paths[h]) < 2:
      return True
    for i in range(len(paths[h])-1):
      a = (paths[h][i], paths[h][i + 1])
      if a in A_s:
        in_service[h] += [k for k in range(paths[h][i][1], paths[h][i + 1][1] + 1)]
      if a in A_g:
        in_refuel[h] += [k for k in range(paths[h][i][1], paths[h][i + 1][1] + 1)]
      if a[1][1] <= e_time and (a in A_s + A_g):
        left_pos[h] = a[1] if a[1][1] >= left_pos[h][1] else left_pos[h]
      if a[0][1] >= e_time and (a in A_s + A_g):
        right_pos[h] = a[0] if a[0][1] <= right_pos[h][1] else right_pos[h]
        if a in A_g:
          next_refuel[h]  = a[0] if a[0][1] <= next_refuel[h][1] else next_refuel[h]
    #new_slot has to be inserted between left_pos and right_pos
    #first, check time condition
    if e_time in in_service[h] or e_time in in_refuel[h]:
      feasibility[h] = False
      continue
    if left_pos[h][1] + fly_time[(left_pos[h][0], leg[0])] > e_time:
      feasibility[h] = False
      continue
    min_time = fly_time[leg] + 14 + fly_time[(leg[1], right_pos[h][0])]
    if right_pos[h][1] - e_time < min_time:
      feasibility[h] = False
      continue

    add_fuel_needed = carac_heli[h]["conso_per_minute"] * (min_time + fly_time[(leg[0], left_pos[h][0])])
    i, j = paths[h].index(left_pos[h]), paths[h].index(next_refuel[h])
    min_refuel_time = min(refuel_times[left_pos[h][0]], refuel_times[right_pos[h][0]], refuel_times[leg[0]], refuel_times[leg[1]])
    for k in range(i, j - 1):
      loc, t = paths[h][k][0], paths[h][k][1]
      a = (paths[h][k], paths[h][k+1])
      if not (a in A_w) and v[h][loc][t].varValue - add_fuel_needed < min_fuel:
        if right_pos[h][1] - left_pos[h][1] < min_time + min_refuel_time:
          feasibility[h] = False
          continue
  return sum(list(feasibility.values())) > 0



def init_slots(locations, colors, r, T):
  """ Initializes all slots. Different status are :
            - booked, if slot is booked
            - bookable, not booked but feasible without breaking current schedule - possible to serve but might increase cost
            - not feasible, not feasible without breaking current schedule
            - revenu opportunity, slot is feasible without increasing current schedule cost (because of deadheads).
      --------------
      Params :
              locations : list, contains all locations
              colors : dict, color for different status
              r : dict, contains all request
              T : list, time steps
      --------------
      Returns :
              slots : dict, contains all slots with initial status

  """

  leg = [(i, j) for i in locations for j in locations if i != j]
  slots = {f: {} for f in leg}
  booked = {f:[] for f in leg}
  for re in r:
    booked[(re[0], re[1])] += [i for i in range(re[-1], re[-1]+5)]
  for f in slots:
    for y in [i for i in range(len(T))][::5]:
      if y in booked[f]:
        slots[f][y] = "booked"
      elif sum([y + i in booked[f] for i in range(-7, 7)]):
         slots[f][y] = "not feasible"
      else:
        slots[f][y] = "bookable"
  return slots


def viz_slots(slots, colors, locations, r, helicopters, T):
  """ Viz function for the booking table, relies on the slots status.
      --------------
      Params :
            slots : dict, output of init_slots() / update_slots()
            colors : dict, gives colors for differents status
            locations : list, contains all locations
            r : dict, contains all requests
            helicopters : list, contains all helicopters
            T : list, contains all time steps

      --------------
      Returns :
            fig : matplotlib fig to be plotted/saved.

  """
  leg = list(slots.keys())
  fig, ax = plt.subplots(figsize=(15, 10))
  ax.set_axis_on()
  ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
  plt.xlabel("Time")
  plt.yticks([i for i in range(1, len(leg)+2)], [l[0] + " to " + l[1]  for l in leg])
  plt.xticks([i for i in range(len(T) + 20)][::15], ['{:02d}:{:02d}'.format(*divmod(870 + i, 60)) for i in range(len(T) + 20)][::15])
  for x in range(1, len(leg) + 1):
    for y in [i for i in range(len(T))][::5]:
      rect = matplotlib.patches.Rectangle((y, x), 4, 0.25, color=colors[slots[leg[x-1]][y]])
      ax.add_patch(rect)
  patchs = []
  for k, v in colors.items():
    patch = mpatches.Patch(color=v, label=k)
    patchs.append(patch)
  plt.legend(handles=patchs)
  plt.title("Booking table. Rand." + str(len(r)) + ".H" + str(len(helicopters)))

  return fig


def viz_fuel(v, paths, carac_heli, T):
  """ Viz funtion for fuel consumption. Plot a graph of fuel level over the service period for each helicopter.
      --------------
      Params :
            v : pulp dict variable, contains fuel levels
            paths : dict, output of get_helicopter_path()
            carac_heli : dict, contains caracteristics for helicopters
            T : list, time steps

      --------------
      Returns :
            fig : matplotlib figure to be plotted/saved.

  """
  fuel_level = {h: [] for h in paths}
  times = {h: [] for h in paths}
  fig, ax = plt.subplots(figsize=(15, 10))
  ax.set_axis_on()
  ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
  for h in paths:
    for pos in paths[h]:
      fuel_level[h].append(100 * v[h][pos[0]][pos[1]].varValue / carac_heli[h]["fuel_cap"])
      times[h].append(pos[1])

  for h in fuel_level:
    plt.plot(times[h], fuel_level[h], label="Fuel level of %s" % h)
  plt.legend(loc="best")
  plt.title("Fuel level during service")
  plt.xticks([i for i in range(len(T) + 20)][::15], ['{:02d}:{:02d}'.format(*divmod(870 + i, 60)) for i in range(len(T) + 20)][::15])
  plt.xlabel("Time")
  plt.ylabel("Fuel level in percent of capacity")

  return fig



def preprocessing(r, fly_time, helicopters, carac_heli, refuel_time, locations, H, len_encod, reverse_indices):
    """ Preprocessing to determine which requests cannot be served by the same helicopter.
        For requests which can be served one after the other, determine :
            - best refuelling spot (used to compute path later on)
            - cost of going from one to the other with and without refuelling (if feasible)
            this includes parking cost if it happens.
        This gives all pairwise pairs (prev, succ) of demands - the idea is that any path can be decompose
        into a list of pairs (prev, succ) of successive requests and therefore the cost can be computed
        before hand. Faster computations in the meta heuristics.
        The case where only one request is served by an helicopter is handled by putting None in place of
        succ.
        -------
        Params :
                r : dict, key is a tuple origin, destination, earliest_dep and value is a list of tuples giving authorized legs.
                fly_time : dict, gives the fly time duration between any two skyports
                helicopters : list containing id of helicopters to use
                carac_heli : dict, contains helicopters' characteristics
                refuel_price : dict, contains refuel price in dollars for each skyports
                refuel_time : dict, contains refuel times in minutes for each skyports
                parking_fees : dict, contains parking fee in $ for each skyport
                locations : list, contains all skyports
                landing_fee : dict, contains landing fee in $ for each skyport
        -------
        Returns :
                 time_compatible: numba dict, contains all pairs of requests that be served one after the other.
                 refuel_compatible: numba dict, contains all pairs of requests that be served one after the other with refuel inbetween.

    """

    #-- Dictionnaries created to be numba-friendly
    time_compatible = numba.typed.Dict.empty(
                key_type=numba.typeof((2,2)),
                value_type=numba.types.int64,
                )
    refuel_compatible = numba.typed.Dict.empty(
                key_type=numba.typeof((2,2)),
                value_type=numba.types.int64,
                )
    for r1 in r:

        for h in helicopters:
            h_i = - int(h.replace('h', ''))
            start = carac_heli[h]["start"]

            if fly_time[(start, r1[0])] <= r1[2]:
                ind = int(reverse_indices[h][r1])
                time_compatible[(h_i, ind)] = 1

            if fly_time[(start, r1[0])] + refuel_time[start] <= r1[2]:
                ind = int(reverse_indices[h][r1])
                refuel_compatible[(h_i, ind)] = 1


        for r2 in r:
            if r1 == r2:
                continue

            first = r1 if r1[2] <= r2[2] else r2
            second = r1 if first == r2 else r2
            if r[first][-1][1][1] + fly_time[(first[1], second[0])] <= second[2]:
                ind_first, ind_second = reverse_indices[helicopters[0]][first], reverse_indices[helicopters[0]][second]
                for i in range(H):
                    time_compatible[(ind_first + len_encod * i, ind_second + len_encod * i)] = 1

                if r[first][-1][1][1] + fly_time[(first[1], second[0])] + refuel_time[first[1]] <= second[2] :
                    for i in range(H):
                        refuel_compatible[(ind_first + len_encod * i, ind_second + len_encod * i)] = 1


    return time_compatible, refuel_compatible

def chain_requests(r, helicopters, carac_heli):
  """ Creates all couples of requests in the right format for MILP2.
      Will be used to compute cost associated to each of this couple.

        Args:
            r : dict, key is a tuple origin, destination, earliest_dep and value is a list of tuples giving authorized legs.
            helicopters : list containing id of helicopters to use
            carac_heli : dict, contains helicopters' characteristics

        Returns:
            couples: list, contains all couples of requests where a request is under format (r1-, r1+, tr1-, tr1+)
            see overleaf section MILP2.

  """
  couples = []
  for h in helicopters:
    start = (carac_heli[h]["start"], carac_heli[h]["start"], 0, 0)
    for req in r:
      couples.append((start, (req[0], req[1], req[2], r[req][-1][1][1])))
      couples.append(((req[0], req[1], req[2], r[req][-1][1][1]), start))
  for r1 in r:
    for r2 in r:
      couples.append(((r1[0], r1[1], r1[2], r[r1][-1][1][1]), (r2[0], r2[1], r2[2], r[r2][-1][1][1])))
  return couples

def cost_chain(couples, helicopters, carac_heli, fly_time, landing_fee, beta=500):
  """ Computes cost of chaining every couple of requests.

            Args:
                couples: list, contains all couples of requests, output of chain_requests
                fly_time : dict, gives the fly time duration between any two skyports
                helicopters : list containing id of helicopters to use
                carac_heli : dict, contains helicopters' characteristics
                beta: float, gives the gain in serving a demand.
                landing_fee : dict, contains landing fee in $ for each skyport

            Returns:

                cost: nested dict, contains cost to chains any pair of requests for each helicopter.
                      cost[h][r1][r2] is the cost for h to serve r2 right after serving r1.

  """
  cost = {h : {p[0]:{} for p in couples} for h in helicopters}
  for h in cost:
    for p in couples:
      if p[0] == p[1]:
        cost[h][p[0]][p[1]] = 0
        continue
      if p[0][2] == p[0][3] == p[1][2] == p[1][3]:
        cost[h][p[0]][p[1]] = 0
        cost[h][p[1]][p[0]] = 0
        continue
      c = fly_time[p[0][1], p[1][0]] * carac_heli[h]["cost_per_min"] + landing_fee[p[1][0]] * (p[1][0] != p[0][1])
      c += fly_time[p[1][0], p[1][1]] * carac_heli[h]["cost_per_min"] + landing_fee[p[1][1]] - beta
      cost[h][p[0]][p[1]] = c
  return cost

def chain_requests_mh(r, helicopters, carac_heli):
      """ DUPLICATE OF CHAIN REQUESTS
          Args:
            r : dict, key is a tuple origin, destination, earliest_dep and value is a list of tuples giving authorized legs.
            helicopters : list containing id of helicopters to use
            carac_heli : dict, contains helicopters' characteristics

          Returns:
            couples: list, contains all couples of requests where a request is under format (r1-, r1+, tr1-, tr1+)
            see overleaf section MILP2.
      """
      couples = []
      for h in helicopters:
        start = (carac_heli[h]["start"], carac_heli[h]["start"], 0)
        for req in r:
          couples.append((start, (req[0], req[1], req[2])))
          couples.append(((req[0], req[1], req[2]), start))
      for r1 in r:
        for r2 in r:
          couples.append(((r1[0], r1[1], r1[2]), (r2[0], r2[1], r2[2])))
      return couples

def chain_requests_mh_nb(r, helicopters, carac_heli):
      """ -- PART OF REFACTORING --
          Duplicate of chain_request but using intergers ids instead of tuples.
          Useful to make MH faster and to use numba.

            Args:
            r : dict, key is a tuple origin, destination, earliest_dep and value is a list of tuples giving authorized legs.
            helicopters : list containing id of helicopters to use
            carac_heli : dict, contains helicopters' characteristics

          Returns:
            couples: list, contains all couples of requests where a request is (r-, r+, tr-, r_id)

      """
      couples = []
      for h_idx, h in enumerate(helicopters):
        start = (carac_heli[h]["start"], carac_heli[h]["start"], 0, -(h_idx+1))
        for r_idx, req in enumerate(r):
          couples.append((start, (req[0], req[1], req[2], r_idx + 1)))
          couples.append(((req[0], req[1], req[2], r_idx), start))
      for r1_idx, r1 in enumerate(r):
        for r2_idx, r2 in enumerate(r):
          couples.append(((r1[0], r1[1], r1[2], r1_idx + 1), (r2[0], r2[1], r2[2], r2_idx + 1)))
      return couples

def cost_chain_mh(couples, helicopters, carac_heli, fly_time, landing_fee, r, parking_fee, beta=500):
      """  Duplicate of cost_chain
            Computes cost of chaining every couple of requests.

            Args:
                couples: list, contains all couples of requests, output of chain_requests
                fly_time : dict, gives the fly time duration between any two skyports
                helicopters : list containing id of helicopters to use
                carac_heli : dict, contains helicopters' characteristics
                beta: float, gives the gain in serving a demand.
                landing_fee : dict, contains landing fee in $ for each skyport

            Returns:

                cost: nested dict, contains cost to chains any pair of requests for each helicopter.
                      cost[h][r1][r2] is the cost for h to serve r2 right after serving r1.

      """
      cost = {h : {p[0]:{} for p in couples} for h in helicopters}
      for h in cost:
        for p in couples:
          if p[0] == p[1]:
            continue
          if p[0][2] > p[1][2]:
            continue
          c = fly_time[p[0][1], p[1][0]] * carac_heli[h]["cost_per_min"] + landing_fee[p[1][0]] * (p[1][0] != p[0][1])
          c += fly_time[p[1][0], p[1][1]] * carac_heli[h]["cost_per_min"] + landing_fee[p[1][1]] - beta
          t2 = p[1][2]
          if p[0][1] == p[0][0]:
            t1 = 0
          else:
            t1 = r[p[0]][-1][1][1]
          free_delta = t2 - t1 - fly_time[(p[0][1], p[1][0])]
          if p[0][1] == p[1][0]:
              c += (free_delta > 15) * parking_fee[p[0][1]]
          else:
              c += (free_delta > 15 * 2) * min(parking_fee[p[0][1]], parking_fee[p[1][0]])
          cost[h][p[0]][p[1]] = c

      return cost

def cost_chain_rf_mh(couples, helicopters, carac_heli, fly_time, landing_fee, r, parking_fee, refuel_time, beta=500):
      """ Computes cost of chaining every couple of requests with refuel inbetween.

            Args:
                couples: list, contains all couples of requests, output of chain_requests
                fly_time : dict, gives the fly time duration between any two skyports
                helicopters : list containing id of helicopters to use
                carac_heli : dict, contains helicopters' characteristics
                beta: float, gives the gain in serving a demand.
                landing_fee : dict, contains landing fee in $ for each skyport

            Returns:

                cost: nested dict, contains cost to chains any pair of requests for each helicopter.
                      cost[h][r1][r2] is the cost for h to serve r2 right after serving r1.

      """
      cost = {h : {p[0]:{} for p in couples} for h in helicopters}
      for h in cost:
        for p in couples:
          if p[0] == p[1]:
            continue
          if p[0][2] > p[1][2]:
            continue
          c = fly_time[p[0][1], p[1][0]] * carac_heli[h]["cost_per_min"] + landing_fee[p[1][0]] * (p[1][0] != p[0][1])
          c += fly_time[p[1][0], p[1][1]] * carac_heli[h]["cost_per_min"] + landing_fee[p[1][1]] - beta
          t2 = p[1][2]
          if p[0][1] == p[0][0]:
            t1 = 0
          else:
            t1 = r[p[0]][-1][1][1]
          free_delta = t2 - t1 - fly_time[(p[0][1], p[1][0])] - refuel_time[p[0][1]]
          if p[0][1] == p[1][0]:
              c += (free_delta > 15) * parking_fee[p[0][1]]
          else:
              c += (free_delta > 15 * 2) * min(parking_fee[p[0][1]], parking_fee[p[1][0]])
          cost[h][p[0]][p[1]] = c

      return cost

def conso_fuel_mh(couples, helicopters, carac_heli, fly_time):
      """ Compute the quantity of fuel needed for each aircraft to server to request
          one after the other.
          Args:
                couples: list, contains all couples of requests, output of chain_requests
                fly_time : dict, gives the fly time duration between any two skyports
                helicopters : list containing id of helicopters to use
                carac_heli : dict, contains helicopters' characteristics
          Returns:
                  conso: nested dict, contains fuel needed to serve any pair of requests for each helicopter.
                      cost[h][r1][r2] is the fuel needed for h to serve r2 right after serving r1.

      """
      conso = {h : {p[0]:{} for p in couples} for h in helicopters}
      for h in conso:
        for p in couples:
          if p[0] == p[1]:
            conso[h][p[0]][p[1]] = 0
            continue
          if p[0][2] == p[1][2]:# == p[1][3]:
            conso[h][p[0]][p[1]] = 0
            conso[h][p[1]][p[0]] = 0
            continue
          c = fly_time[p[0][1], p[1][0]] * carac_heli[h]["conso_per_minute"]
          c += fly_time[p[1][0], p[1][1]] * carac_heli[h]["conso_per_minute"]

          conso[h][p[0]][p[1]] = c

      return conso






def get_quant(a, q):
    """ Returns quantile of order 1 - q

        Args:
            a: list, or list like contains float representing a PDF.
            q: float, between 0 and 1.
        Returns:
            quantile of order 1-q
    """
    for t in range(len(a)):
        if np.sum(a[t:]) <= q:
            return t - 1

def normal_pdf(mu, sigma, bins):
    """ Returns the PDF of a normal distribution N(mu, sigma) over a discrete support : bins

        Args:
            mu: float, mean of the distribution
            sigma: float, standard deviation of the distribution
            bins: np.ndarray, bins over which to compute the PDF.
                  Typically it is computed using np.linspace()
        Returns:
            y: np.ndarray, PDF of N(mu, sigma)
    """
    y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2))
    return y

def mixture(mu1, mu2, mu3, sig1, sig2, sig3, bins, w1=0.33, w2=0.33, w3=0.34):
    """ Returns the PDF of a mixture of normal distributions over a discrete support bins:
                w1 * N(mu1, sig1) + w2 * N(mu2, sig2) + w3 * N(mu3, sig3)

        w1, w2, w3 must be such that w1 + w2 + w3 = 1

        Args:
            mu1: float, mean of the first normal distribution
            mu1: float, mean of the second normal distribution
            mu1: float, mean of the third normal distribution
            sig1: float, standard deviation of the first normal distribution
            sig2: float, standard deviation of the second normal distribution
            sig3: float, standard deviation of the third normal distribution
            bins: np.ndarray, bins over which to compute the PDF.
                  Typically it is computed using np.linspace()
            w1: float, weight for the first normal distribution
            w2: float, weight for the second normal distribution
            w3: float, weight for the third normal distribution

        Returns:
            y: np.ndarray, the PDF of a mixture of normal distributions.
    """
    y = w1 * normal_pdf(mu1, sig1, bins) + w2 * normal_pdf(mu2, sig2, bins) + w3 * normal_pdf(mu3, sig3, bins)
    return y

def new_demand(i, distri_new_demands, p_type, T):
    """ Simulates a new individual demand. As explained in the overleaf
        these a simulated using a higher level distribution (distri_new_demands),
        The process is to draw a value from T according to distri_new_demand which will
        give the mu of the new demand, i.e. its ETA.
        Then the standard deviation of the new demand is drawn at random.
        This mu and std will describe the normal distribution that models the arrival time of
        the simulated demand.

        Other characteristic are drawn at random.

      Args:
          i: int, id of new demand
          distri_new_demands: np.ndarray, PDF for the likelihood of new demands appearing over time.
          p_type: float, probability of demand being of type 1 (cannot arrive late)
          T : list, time steps

      Returns:
          new_mu: float, mean of normal distribution describing arrival time of i
          new_std: float, standard deviation of normal distribution describing arrival time of i
          new_lbda: float, lambda parameter for the aversion function
          new_type: int, 0 or 1, type of new demand i
          npax[0]: int, number of pax in this demand
          i: int, ID of the new demand

    """
    npax = np.random.choice([1, 2, 3, 4, 5, 6], size = 1, p = np.array([0.7, 0.1, 0.1, 0.07, 0.02, 0.01]))
    new_lbda = 0.1
    new_type = np.random.choice([0, 1], p=[1-p_type, p_type])
    new_mu = np.random.choice(T, p=distri_new_demands)
    new_std = np.random.choice([7, 10, 5])
    return new_mu, new_std, new_lbda, new_type, npax[0], i

def generate_ind_demands(n, distri_new_demands, p_type, T, deltas):
    """ This function uses the new_demand function to generate n independent demand.
        It also compute the quantiles associated with each distribution.


        Args:
            n: int, number of demands to generate
            distri_new_demands: np.ndarray, distribution of new demands over time. Gives the likelihood
                                of new demands appearing over time. Here it will directly simulate the
                                ETA (mean of arrival times) at origin skyport.
            p_type: float, between 0 and 1. Gives the probability of being of type 1, i.e. hard constraint on arrival time
                    at destination.
            T: list, time steps of service period.
            deltas: dict, gives the maximum allowed probability of being late associated to each type of demand.

        Returns:
            points: list, contains tuple which are output of the new_demand function
            pt: list, contains id of every demand generated
            costs: dict, contains the "cost" of pooling any two demand together, by computing the
                  difference between their quantile. This will be passed through the aversion function.
            weights: dict, contains for each demand the number of pax in it, useful for the ILP
            quants: dict, contains for each demand its quantile depending on its type.

    """
    points = []
    pt = []
    for i in range(n):
        new_mu, new_std, new_lbda, new_type, npax, j = new_demand(i, distri_new_demands, p_type, T)
        points.append((new_mu, new_std, new_lbda, new_type, npax, j))
        pt.append(i)
    quantiles = []
    for p in points:
        a = normal_pdf(p[0], p[1], np.linspace(0, 210, 210))
        quantiles.append(get_quant(a, deltas[p[3]]))
    weights = {}
    costs = {}
    quant = {}
    for i in range(len(points)):
        weights[i] = points[i][-2]
        for j in range(len(points)):
            costs[(i, j)] = abs(quantiles[i] - quantiles[j])
        quant[i] = quantiles[i]
    return points, pt, costs, weights, quant


def get_groups(match, points):
    """ DUPLICATE FUNCTION, in src/utils_pooling.py
        should be ignored
    """
    G = {k: [] for k in match.groups}
    F = {k:0 for k in match.groups}
    for i in match.points:
        for k in match.groups:
            if match.x[i][k].varValue == 1:
                G[k].append(points[i])
                if match.quant[i] >= F[k]:
                    F[k] = match.quant[i]
    return G, F

def plot_groups(groups, flights, T, points, l):
    """DUPLICATE found in src/utils_pooling.py
    """

    colors=cm.rainbow(np.linspace(0,1, len(groups)))
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.title("Pooled ETA on leg " + str(l) + " | " + str(len(points)) + " random requests.")
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel("Time")

    for i in groups:
        for d in groups[i]:
            ax.plot(np.linspace(0, max(T), max(T)), normal_pdf(d[0], np.sqrt(d[1]), np.linspace(0, max(T), max(T))), label = str(i), color = colors[i], zorder = -10)

        ax.add_patch( Rectangle((flights[i] - 5, 0.),
                            5, 0.0,
                            fc ='none',
                            ec = colors[i],
                            fill = True,
                            lw = 8))

    plt.xticks([i for i in range(len(T))][::15], ['{:02d}:{:02d}'.format(*divmod(870 + i, 60)) for i in range(len(T))][::15])
    plt.ylabel("ETA likelihood")
    plt.show()


def compute_exp_waiting(groups, flights, points, T):
    """ Computes expected waiting time of passengers given groups and flights created.

        Params :
                groups: dict, contains for all flights the list of assigned demand IDs
                flights: dict, contains for all flights, the departure time.
                T: list, time steps of the service period.
                points: list, contains all individual demands under format new_demand().

        Returns :
                W: dict, contains expected waiting time for each demand.
    """
    W = {}
    T = np.linspace(0, 210, 210)
    for g in groups:
        for d in groups[g]:
            #compute expected waiting time
            dens = normal_pdf(d[0], np.sqrt(d[1]), T)
            if np.sum(dens) != 0:
              dens = dens / np.sum(dens)
            start = flights[g]
            expw = np.dot(np.flip(T[:start - 5]), dens[:start - 5])
            W[d] = expw
    return W


def numba_dict(r, fly_time, helicopters, carac_heli, refuel_time, locations, refuel_price,
                  parking_fee, landing_fee, beta):
    """ PART OF REFACTORING
        Translating some of the existing parameters for optim to numba compliant types.

        Params :
                r : dictionnary containing requested flights
                helicopters : list containing id of helicopters to use
                carac_heli : dict, contains helicopters' characteristics
                refuel_time : dict, contains refuel times in minutes for each skyports
                refuel_price: dict, contains refuel price in dollars for each skyports
                locations : list, contains all skyports
                parking_fees : dict, contains parking fee in $ for each skyport
                landing_fees : dict, contains landing fee in $ for each skyport
                fly_time : dict, gives the fly time duration between any two skyports
                beta : int, give the gain in serving a demand in $. Constant for now, could be demand-dependent in the future.

        Returns :
                memo_cost_nb: dict numba typed empty, will contain cost for memoization
                cost_rf_nb: dict numba typed, contains cost for any heli to start by serving a request with refuel inbetween
                chain_rf_nb: dict numba typed, contains cost for any heli to chain two requests with refuel inbetween
                chain_nb: dict numba typed, contains cost for any heli to chain two requests
                req_origin: dict numba typed, contain for each request its origin
                req_dest: dict numba typed, contains for each request its destination
                fly_time_nb: dict numba typed, contains fly times


    """

    #-- Dictionnaries created to be numba-friendly
    couples = chain_requests_mh_nb(r, helicopters, carac_heli)
    tup_example = [2] * (2 * len(r) // 64 + 1)
    tup_example = tuple(tup_example)
    memo_cost_nb = numba.typed.Dict.empty(
                key_type = numba.types.int64,
                value_type = numba.types.float64[:],
                )
    cost_rf_nb = numba.typed.Dict.empty(
                key_type=numba.typeof((2, 2)),
                value_type= numba.typeof(float(1)),
                )
    for h_idx, h in enumerate(helicopters):
      for i, req in enumerate(r):
        cost_rf_nb[(h_idx, i+1)] = float(refuel_price[req[1]])
      cost_rf_nb[(h_idx, -(h_idx+1))] = float(refuel_price[carac_heli[h]['start']])

    chain_rf_nb = numba.typed.Dict.empty(
                key_type=numba.typeof((2,2,2)),
                value_type= numba.types.float64,
                )
    chain_nb = numba.typed.Dict.empty(
                key_type=numba.typeof((2,2,2)),
                value_type= numba.types.float64,
                )

    for h_idx, h in enumerate(helicopters):
      for p in couples:
          if p[0] == p[1]:
            continue
          if p[0][2] > p[1][2]:
            continue
          c = fly_time[p[0][1], p[1][0]] * carac_heli[h]["cost_per_min"] + landing_fee[p[1][0]] * (p[1][0] != p[0][1])
          c += fly_time[p[1][0], p[1][1]] * carac_heli[h]["cost_per_min"] + landing_fee[p[1][1]] - beta
          t2 = p[1][2]
          if p[0][1] == p[0][0]:
            t1 = 0
          else:
            t1 = r[(p[0][0], p[0][1], p[0][2])][-1][1][1]
          free_delta1 = t2 - t1 - fly_time[(p[0][1], p[1][0])] - refuel_time[p[0][1]]
          free_delta2 = t2 - t1 - fly_time[(p[0][1], p[1][0])]

          if p[0][1] == p[1][0]:
              c1 = c + (free_delta1 > 15) * parking_fee[p[0][1]]
              c2 = c + (free_delta2 > 15) * parking_fee[p[0][1]]
          else:
              c1 = c + (free_delta1 > 15 * 2) * min(parking_fee[p[0][1]], parking_fee[p[1][0]])
              c2 = c + (free_delta2 > 15 * 2) * min(parking_fee[p[0][1]], parking_fee[p[1][0]])

          chain_rf_nb[(h_idx, p[0][-1], p[1][-1])] = c1
          chain_nb[(h_idx, p[0][-1], p[1][-1])] = c2




    req_origin = numba.typed.Dict.empty(
                key_type=numba.typeof(2),
                value_type= numba.types.unicode_type,
                )


    req_dest = numba.typed.Dict.empty(
                key_type=numba.typeof(2),
                value_type= numba.types.unicode_type,
                )
    for req_idx, req in enumerate(r):
      req_origin[req_idx+1] = req[0]
      req_dest[req_idx+1] = req[1]

    fly_time_nb = numba.typed.Dict.empty(
                key_type=numba.typeof(('S1', 'S2')),
                value_type= numba.types.int64,
                )

    for k, v in fly_time.items():
      #print(k)
      fly_time_nb[k] = int(v)

    return memo_cost_nb, cost_rf_nb, chain_rf_nb, chain_nb, req_origin, req_dest, fly_time_nb


@njit
def hash_sol(sol, min_x, max_x, h_idx, list_nb, k):
  """ Hash function for a solution part corresponding to h.

      Args:
            sol: np.ndarray, solution
            min_x: int, min index corresponding to h in sol
            max_x: int, max index corresponding to h in sol
            h_idx: int, idx - id of h
            list_nb: list numba typed empty can contains int
            k: int, bits for encoding

      Returns:
            list_nb: list numba typed, contains hash.


  """
  list_nb[0] = h_idx
  p = 0
  val = 0
  j = 1
  for i in range(min_x, max_x+1):
    if p == k:
      j += 1
      p = 0
    val += 2 ** p * sol[i]
    list_nb[j] = val
    p += 1
  return list_nb


