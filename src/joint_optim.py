import numpy as np
from numba import jit, njit
import numba
import utils_pooling as up
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import operator
import utils as U
from itertools import chain
import multiprocessing as mp
import itertools
from multiprocessing import Queue, Process
import optim
import time

def sim_data(T, legs, n, seed=9):
  """ Simulates data for joint matchingm i.e. a set of n individual demands and stores their characteristic in a table

      Args:
          T: list, contains all time steps
          legs: list, contains all legs
          n: int, number of demands to be simulated
          seed: int, for random generations

      Returns:
          data: dict, keys are legs values are dict which contains :
                      tab_points : np.ndarray of all individual demand with carac
                      idx: list, ids of demands

  """
  np.random.seed(seed)
  N = len(legs)
  bins = np.linspace(0, len(T), len(T))
  data = {}
  for i, l in enumerate(legs):
      space = [k for k in range(int(len(T[5:-30]) / (N)) * (i), int(len(T[5:-30]) / N) * (i + 1)) if k % 15 == 0]
      mu1, mu2 = np.random.choice(space, 2, replace=False)

      sig1, sig2  = np.random.randint(50, 80, 2)

      dnp = up.mixture(mu1, mu2, 47, sig1, sig2, 40, bins, w1=0.33, w2=0.33, w3=0)
      plt.plot(dnp)
      dnp = dnp / np.sum(dnp)
      deltas = {0: 0.1, 1: 0.2}
      points, ids, costs, weights, quants = up.generate_ind_demands(n, dnp, 0.2, T, deltas)
      q = np.array(list(quants.values())).reshape(-1,1)
      idx = list(range(n)) #maybe order such that conflict are max
      tab_points = np.array(points)
      tab_points = np.concatenate((tab_points, q), axis=1)
      data[l] = {'tab_points': tab_points, 'idx': idx}
  return data

def aversion(times, lbdas, threshold):
    """ Aversion function. See overleaf for motivation about this.

        Args:
            times: np.ndarray, contains arguments for aversion fucntion
            lbdas: np.ndarray, lambda params
            threshold: int, activation for the aversion to start counting.

        Returns:
            float, aversion value

    """
    return (np.exp(np.clip(lbdas * times, 0, 200)) - np.exp(np.clip(lbdas * threshold, 0, 200))) * (times > threshold)


def check_spread(g, tab_points, fq, fp, delta, aversion, threshold):
    """ Checks if the spread constraint is valid within group g.

        Args:
            g: list, group of users pooled together. contains their ids
            tab_points: np.ndarray, contains ind demand caracs
            fq: int, colunm index in tab poitns for quantiles
            fp: int, column index in tab points for params lbda
            delta: int, max aversion spread
            aversion: function, aversion function
            threshold: float, parameter for aversion fucntion

        Returns:
            bool: True iff constraint is ok

    """
    slice_q = tab_points[g, fq]
    slice_params = tab_points[g, fp]
    max_q = np.max(slice_q) #could use the fact that we can keep things ordered here.
    time_diff = max_q - slice_q
    av = aversion(time_diff, slice_params, threshold)
    viol = np.any(av > delta)
    return not (viol)

def check_cap(g, C, tab_points, f):
    """ Checks if the capacity constraint is valid within group g.

        Args:
            g: list, group of users pooled together. contains their ids
            C: int, capacity of one aircraft
            tab_points: np.ndarray, contains ind demand caracs
            f: int, colunm index in tab poitns for pax

        Returns:
            bool: True iff constraint is ok

    """
    return np.sum(tab_points[g, f]) <= C


class PARTITION_NODE():
    """
        Node component for the partition tree search.
        This object models a node in a Tree. See overleaf for more details.
        A node has :
              - a content which is a subpartition in our case
              - a set of children which are also nodes
              - is feasible or not.

    """
    def __init__(self, cont, C, tab_points, fq=-1, fp=4, delta=15, aversion=aversion, threshold=10):
        """ Init the node

          Args:
              cont: list, subpartition
              C: int, capacity of one aircraft
              tab_points: np.ndarray, contains ind demand caracs
              fq: int, colunm index in tab poitns for quantiles
              fp: int, column index in tab points for params lbda
              delta: int, max aversion spread
              aversion: function, aversion function
              threshold: float, parameter for aversion fucntion

          Returns:
              bool: True iff constraint is ok

        """
        self.content = cont
        self.children = []
        #checking whether this node is feasible and is allowed to have children
        feas = True
        for g in cont:
            if len(g) > C:
                feas = False
                break
            #2 is for params lbda
            feas_spread = check_spread(g, tab_points, fq, 2, delta, aversion, threshold)
            if not(feas_spread):
                feas = False
                break
            feas_cap = check_cap(g, C, tab_points, fp)
            if not(feas_cap):
                feas = False
                break
        self.feasible = feas
        self.has_children = False

    def add_child(self, child):
        """ Adds a child
            and updates parents status.
            Args:
                child: instance of PARTION_NODE

            Returns :
                None

        """
        self.children.append(child)
        self.has_children = True




class TREE_SEARCH():
    """
        This class implement the tree search heuristic presented in the overleaf to explore the set of partition.
        It will build a set of admissible minimum size (max pooling) partition.
    """
    def __init__(self, C, tab_points, ordered_idx, delta=15, aversion=aversion, fp=4, fq=-1, threshold=10, parent_buffer=1000):
        """ Init the class

            Args:

                C: int, capacity of one aircraft
                tab_points: np.ndarray, contains ind demand caracs
                ordered_idx: list, contains all demand id, in the order you want to insert them in the tree
                fq: int, colunm index in tab poitns for quantiles
                fp: int, column index in tab points for params lbda
                delta: int, max aversion spread
                aversion: function, aversion function
                threshold: float, parameter for aversion function
                parent_buffer: int, number of parent to keep during search. See overleaf to understand this one
                                it's the K parameter.

            Returns :


        """
        self.C = C
        self.tab_points = tab_points
        self.ordered_idx = ordered_idx
        self.n_points = len(ordered_idx)
        self.delta = delta
        self.aversion = aversion
        self.fp = fp
        self.fq = fq
        self.threshold = threshold
        self.partition_counter_setter()
        self.partition_cache_setter()
        self.min_size = self.n_points
        self.current_size = self.n_points
        self.parent_buffer = parent_buffer

    def is_partition(self, node):
        """ Checks is a node is actually a partition.

            Args :
                  node: instance of PARTITION_NODE

            Returns :
                  bool: true iff node content is a partition.


        """
        flat_content = list(chain.from_iterable(node.content))
        return self.n_points == len(flat_content)

    def check_cap(self, g, C, tab_points, f):
        """ Checks if the capacity constraint is valid within group g.

              Args:
                  g: list, group of users pooled together. contains their ids
                  C: int, capacity of one aircraft
                  tab_points: np.ndarray, contains ind demand caracs
                  f: int, colunm index in tab poitns for pax

              Returns:
                  bool: True iff constraint is ok

        """
        return np.sum(tab_points[g, f]) <= C

    def check_spread(self, g, tab_points, fq, fp, delta, aversion, threshold):
        """ Checks if the spread constraint is valid within group g.

            Args:
                g: list, group of users pooled together. contains their ids
                tab_points: np.ndarray, contains ind demand caracs
                fq: int, colunm index in tab poitns for quantiles
                fp: int, column index in tab points for params lbda
                delta: int, max aversion spread
                aversion: function, aversion function
                threshold: float, parameter for aversion fucntion

            Returns:
                bool: True iff constraint is ok

        """
        slice_q = tab_points[g, fq]
        slice_params = tab_points[g, fp]
        max_q = np.max(slice_q) #could use the fact that we can keep things ordered here.
        time_diff = max_q - slice_q
        av = aversion(time_diff, slice_params, threshold)
        viol = np.any(av > delta)
        return not(viol)

    def create_child(self, parent_node, new_point):
        """ For a given parent node in the tree, this function will create it's set of child when insert
            new_point in the tree

            Args :
                    parent_node: instance of PARTITION_NODE
                    new_point: int, new point id

            Returns :
                    children: list, contains instances of PARTITION_NODE

        """
        children = []
        for parent_id, parent in enumerate(parent_node.content):
            aug_set = parent.copy()
            #rest is the set without current parent which will get augmented by new_point
            rest = parent_node.content[:parent_id] + parent_node.content[parent_id+1:]
            aug_set.append(new_point)
            children.append(PARTITION_NODE(rest + [aug_set],
                                           self.C,
                                           self.tab_points,
                                           self.fq,
                                           self.fp,
                                           self.delta,
                                           self.aversion,
                                           self.threshold))
        children.append(PARTITION_NODE(parent_node.content + [[new_point]],
                                       self.C,
                                       self.tab_points,
                                       self.fq,
                                       self.fp,
                                       self.delta,
                                       self.aversion,
                                       self.threshold))
        return children

    def explore(self, verbose=True, eps=0.15):
        """ This function will build the tree, constructing a set of valid paritions.

            Args :
                  verbose: bool, whether to progress
                  eps: float, exploration probablity when selecting parents.

            Returns :
                  root: instance of PARTITION_NODE, which contains the entire tree built here.

        """
        if len(self.ordered_idx) == 1:
            return self.ordered_idx
        root = PARTITION_NODE([[self.ordered_idx[0]]],
                               self.C,
                               self.tab_points,
                               self.fq,
                               self.fp,
                               self.delta,
                               self.aversion,
                               self.threshold)


        parent_nodes = [root]
        new_parents = []
        parents_bis = []
        heuristic_idx = []
        for t, i in enumerate(self.ordered_idx[1:]):
            if verbose:
                print("-----")
                print(f"Insertion step {t}, inserting {i}")
            new_point = i
            new_parents.clear()
            parents_bis.clear()
            heuristic_idx.clear()


            for parent in parent_nodes: #iterate over fringe nodes

                self.current_size = self.n_points

                children = self.create_child(parent, new_point)
                j = 0
                for child in children:
                    parent.add_child(child)
                    if child.feasible:
                        heuristic_idx.append((len(child.content), j))
                        if len(parents_bis) < self.parent_buffer:
                          u = np.random.random()
                          if u <= eps:
                            parents_bis.append(child)
                        new_parents.append(child)
                        j = j + 1
                        if self.is_partition(child):
                          if len(child.content) == self.min_size:
                              self.partition_cache.append(child.content)
                          elif len(child.content) < self.min_size:
                              self.min_size = len(child.content)
                              self.partition_cache.clear()
                              self.partition_cache.append(child.content)

            heuristic_idx.sort()

            new_parents = [new_parents[e[1]] for e in heuristic_idx[:self.parent_buffer]] + parents_bis
            parent_nodes = new_parents.copy()
        return root

    def partition_counter_setter(self):
        """ Setter for a counter which will counts the number of partition explored.

        """
        self.n_partition = 0

    def partition_cache_setter(self):
        """ Setter for two lists:
                - partition_cache, will store partition obtained during explore
                - partitions sizes, will store the size of these

        """
        self.partition_cache = []
        self.partition_sizes = []

    def dfs(self, node, indent=0):
        """ Depth First Search exploration of a tree built in self.explore
            Will update :
                self.n_partition
                self.partition_cache
                self.partition_sizes
            As the search progresses.

            Args:
                  node: instance of PARTITION_NODE

            Returns:
                  None

        """
        if self.is_partition(node):
            self.n_partition += 1
            if len(node.content) == self.min_size and node.feasible:
                self.partition_cache.append(node.content)
            elif len(node.content) < self.min_size and node.feasible:
                self.min_size = len(node.content)
                self.partition_cache.clear()
                self.partition_cache.append(node.content)

            self.partition_sizes.append(len(node.content))
        for child in node.children:
            self.dfs(child, indent+3)

def get_groups(partition, tab_points, fq):
        """ Extract groups and flights schedule from a partition

            Args:
                parition: list of list forming a partition of the demands id.
                tab_points: np.ndarray, contains all demands characteristics.
                fq: int, column index in tab_point where quantiles are present.

            Returns:
                G: dict, contains all group assigned to their flight (key)
                F: dict, contains flights created with departure date

        """
        G = {}
        F = {}
        for i in range(len(partition)):
            G[i] = partition[i]
            F[i] = np.max(tab_points[partition[i], fq])
        return G, F

def plot_groups(groups, flights, points, T, tab_points):
        """ Plot the pooling

            Args:
                G: dict, contains all group assigned to their flight (key)
                F: dict, contains flights created with departure date
                points: list equivalent of tab_points
                T: list, contains all time steps of service period
                tab_points: np.ndarray, contains all demands characteristics.


            Returns:
                fig: matplotlib figure

        """
        colors=cm.rainbow(np.linspace(0,1, len(groups)))
        fig, ax = plt.subplots(figsize=(15, 7))
        plt.title(f"Pooled ETA on leg L {len(points)} random requests.")
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlabel("Time")

        for i in groups:
            for d in groups[i]:
                mu = tab_points[d, 0]
                sig = tab_points[d, 1]
                ax.plot(np.linspace(0, len(T), len(T)), up.normal_pdf(mu, np.sqrt(sig), np.linspace(0, len(T), len(T))), label = str(i), color = colors[i], zorder = -10)

            ax.add_patch( Rectangle((flights[i] - 5, 0.),
                                5, 0.0,
                                fc ='none',
                                ec = colors[i],
                                fill = True,
                                lw = 8))

        plt.xticks([i for i in range(len(T))][::30], ['{:02d}:{:02d}'.format(*divmod(420 + i, 60)) for i in range(len(T))][::30])
        plt.ylabel("ETA likelihood")
        return fig

def process_partitions(search, tab_points):
    """ Process the search to return a set of possible schedule

          Args:
                search: instance of TREE_SEARCH()
                tab_points: np.ndarray, contains all demands characteristics.

          Returns:

                unique_flights: list, contains tuples of departure date forming flight schedule on one leg.

    """
    all_flights = []
    for part in search.partition_cache:
        G, F = get_groups(part, tab_points, -1)
        all_flights.append(tuple(sorted(list(F.values()))))

    unique_flights = list(set(all_flights))
    return unique_flights


def pool_all(data):
    """ Perform pooling over all legs

        Args:
            data: dict, from function sim_data.

        Returns:
            possible_set: dict, contains for each leg (key), the unique_flights schedule possible (list containing different possibilities)
            nb_possible: dict, for each leg (key) give the number of possible schedule found.

    """
    possible_set = {}
    nb_possible = {}
    for leg in data:
        search = TREE_SEARCH(4, data[leg]["tab_points"], data[leg]["idx"])
        tree = search.explore(verbose=False)
        possible_flights = process_partitions(search, data[leg]["tab_points"])
        possible_set[leg] = possible_flights
        nb_possible[leg] = len(possible_flights)
    # --- possible set contains for each the possible dates of flights to be scheduled
    return possible_set, nb_possible

def create_demand(possible_set, inds, fly_time, verbose=False):
    """ Will create the requested flights to feed to the routing optimizer from the pooling results.


        Args:
            possible_set: dict, contains for each leg (key), the unique_flights schedule possible (list containing different possibilities)
            inds: dict, give the index of which possible schedule to take from possible set for each leg.
            fly_time: dict, give fly time between legs

        Returns:
            R: request dict as in src/optim.META_HEURISTIC

    """
    requested_flights = {}
    for leg in inds:
        ori, dest = leg[0], leg[1]
        for t in possible_set[leg][inds[leg]]:
            if t > 680:
                t = 680
            e_time_str = '{:02d}:{:02d}'.format(*divmod(420 + int(t), 60))
            e_time = int(t)
            if verbose:
                print(f"Flight at {t} on leg {leg} with earliest departure time at {e_time_str}.")
            #----Now translate flights in MH format with time window.
            requested_flights[(ori, dest, np.int64(e_time))] = []
            for w in range(5):
                requested_flights[(ori, dest, np.int64(e_time))].append(((ori, np.int64(e_time + w)), (dest, np.int64(e_time + w + 14 + fly_time[(ori, dest)]))))
    #-- sorting to avoid errors in MH.
    R = {}
    keys_sorted = sorted(list(requested_flights.keys()), key=lambda x: x[2])
    for k in keys_sorted:
        R[k] = requested_flights[k]
    return R


def move_partition_heuri(ind, possible_set, tabu):
    """
        Heuristic to select other partition after receiving feed from MH.
        Ideas : if served_mh == 1 : stop
        else:
        make one move, i.e. change pooling for one leg by choosing the one that it further away
        from the previously used one. (measuring L2 distance between departure dates for flights)

        Args:
            possible_set: dict, contains for each leg (key), the unique_flights schedule possible (list containing different possibilities)
            inds: dict, give the index of which possible schedule to take from possible set for each leg.
            tabu: dict, give forbidden move at this stage.

        Returns:
            bool: True iff a move could be made.
            new_inds: updated inds dict
            tabu: updated tabu dict


    """
    max_delta = 0
    leg_max = None
    new_inds = ind.copy()
    for leg in possible_set:
        if len(possible_set[leg]) == 1:
            new_inds[leg] = ind[leg]
            continue
        #otherwise look for non tabu max delta move
        current_schedule = possible_set[leg][ind[leg]]
        for i, schedule in enumerate(possible_set[leg]):
            if i == ind[leg] or i in tabu[leg]:
                continue
            else:
                delta = np.linalg.norm(np.array(schedule) - np.array(current_schedule))
                if max_delta < delta:
                    max_delta = delta
                    max_ind = i
                    leg_max = leg
    if not(leg_max):
        print("No move to make")
        return False, ind, tabu
    new_inds[leg_max] = max_ind
    print(f"Move to make on leg {leg_max} - distance {max_delta}.")
    tabu[leg_max].append(max_ind)
    return True, new_inds, tabu



def sim_common_data(l, h, seed=9):
  """ Simulate infrastructure data that can be shared between joint optim and sequential optim.
      SEE src/optim.META_HEURISTIC for more details about what is generated here

      Args:
            l: int, number of skyports
            h: int, number of helicopters in fleet
            seed: int for random generation

      Returns:
            dict:
                helicopters: list
                skyports: list
                refuel_times: dict
                refuel_prices: dict
                T: list,
                fly_time: dict
                fees: dict, landing fees
                parking_fee: dict, parking fees
                beta: float, gain in serving one request
                min_takeoff: float, minimum level of fuel to takeoff
                n_iter: int, iteration max for VNS in metaheuristic
                pen_fuel: float, fuel penalty, should be 0
                no_imp_limit: int, parameter for VNS in metaheuristic
                nodes: list, nodes for time space network
                carac_heli: dict


  """

  np.random.seed(seed)
  helicopters = [f"h{i}" for i in range(1, h+1)]
  skyports = [f"S{i}" for i in range(1, l + 1)]
  refuel_times = {loc: np.random.choice([25, 15, 5], p=np.array([0.5, 0.3, 0.2])) for loc in skyports}
  prices = {25: 100, 15: 200, 5: 700}
  refuel_prices = {loc: prices[refuel_times[loc]] for loc in skyports}
  T = [k for k in range(720)]  # timesteps
  fly_time = {}
  for s1, s2 in itertools.combinations(skyports, r = 2):
      fly_time[(s1, s2)] = np.random.choice([7, 10, 15])
      fly_time[(s2, s1)] = fly_time[(s1, s2)]
  for s in skyports:
      fly_time[(s, s)] = 0
  fees = {loc: np.random.choice([200, 300, 100]) for loc in skyports}
  parking_fee = {loc : np.random.choice([200, 300, 100, 0], p=np.array([0.5, 0.1, 0.3, 0.1])) for loc in skyports}

  # gain from serving a request, arbitrary for now but will depend on number
  # of passengers and prices charged.
  beta = 500
  min_takeoff = 325
  n_iter = 10000
  pen_fuel = 0
  no_imp_limit = 1600
  # nodes for the time space network

  nodes = list(itertools.product(skyports, T))

  carac_heli = {h : {
          "cost_per_min": 34,
          "start": np.random.choice(skyports),
          "security_ratio": 1.25,
          "conso_per_minute": 20,
          "fuel_cap": 1100,
          "init_fuel": np.random.choice([900, 1100, 1000]),
          "theta": 1000}  for h in helicopters}
  return {"helicopters":helicopters,
                "skyports":skyports,
                "refuel_times":refuel_times,
                "refuel_prices":refuel_prices,
                "T": T,
                "fly_time":fly_time,
                "landing_fee":fees,
                "parking_fee":parking_fee,
                "beta":beta,
                "min_takeoff":min_takeoff,
                "n_iter":n_iter,
                "pen_fuel":pen_fuel,
                "no_imp_limit":no_imp_limit,
                "nodes":nodes,
                "carac_heli": carac_heli}



def MH_probe(requests, common_data, seed, process_q):
    """ Runs one instance of the MH and update the multiprocessing Queue
        Used to run MH in parralel

        Args:
            requests: dict, set of request to serve by MH
            common_data: dict, output of sim_common_data
            seed: int, for random number gen
            process_q: multiprocessing queue

        Returns:
            None


    """
    print(f"Running MH probe with seed {seed}")
    np.random.seed(seed)
    # -- updating arcs
    A_w, A_s, A_f, A_g = U.create_arcs(common_data['nodes'],
                                       requests,
                                       common_data['fly_time'],
                                       common_data['refuel_times'])
    A = A_w + A_s + A_f + A_g
    # -- now run MH
    meta = optim.META_HEURISTIC(
                        requests,
                        common_data['helicopters'],
                        common_data['carac_heli'],
                        common_data['refuel_times'],
                        common_data['refuel_prices'],
                        common_data['skyports'],
                        common_data['nodes'],
                        common_data['parking_fee'],
                        common_data['landing_fee'],
                        common_data['fly_time'],
                        common_data['T'],
                        common_data['beta'],
                        common_data['min_takeoff'],
                        common_data['pen_fuel'],
                        A_s,
                        A,
                        A_g)

    meta.init_encoding()
    meta.init_compatibility()
    meta.init_request_cost()
    init_heuri = meta.init_heuristic_random()
    best_sol, best_cost, _, _, perf_over_time, _, text_log = meta.VNS(
        init_heuri, common_data['n_iter'], eps=0.12, no_imp_limit=common_data['no_imp_limit'], verbose=0)
    unserved, served = meta.update_served_status(best_sol)
    fuel_check = meta.fuel_checker(best_sol)
    print(f"{seed} finished, caching best metrics.")

    #putting results into the queue
    process_q.put((best_sol, best_cost, round(len(served) / len(meta.r), 3), fuel_check))



def runInParallel(seeds, MH_probe, R, common_data):
    """ runs in paralel several instance of the MH

          Args:
                MH_probe: function
                R: set of requests
                common_data: dict, output of sim_common data.

          Returns:
              pqueue: multiprocessing queue

    """
    pqueue = Queue(maxsize=len(seeds))
    proc = []
    for s in seeds:
        p = Process(target=MH_probe, args=(R, common_data, s, pqueue))
        proc.append(p)
        p.start()
    for p in proc:
        p.join()

    return pqueue





def heuristic_search(common_data, possible_set, legs, n_iter):
    """ Seach over the cartesian product of different pooling over legs to find which is best.
        See overleaf for more details about this.

        Args :
              common_data: dict, from sim_common_data()
              possible_set: dict, contains for each leg (key), the unique_flights schedule possible (list containing different possibilities)
              legs: list, contains all legs
              n_iter: max number of iteration for the search.


        Returns :
              best_element:
              logs: list of list containing candidate encountered during search.


    """
    print("Starting search....")
    inds = {leg: 0 for leg in legs}
    tabu = {leg: [0] for leg in legs}
    # -- creating demands for MH
    R = create_demand(possible_set, inds, common_data['fly_time'], verbose=False)

    # -- Start probing this pooling in parralel
    start = time.time()
    elements = list()
    samples = [s for s in range(6, 15)]
    q = runInParallel(samples, MH_probe, R, common_data)
    for i in range(len(samples)):
        elements.append(q.get())
    print(f"Got all results in {round(time.time() - start, 2)} seconds.")
    best_element = sorted(elements, key=lambda x: - x[2])[0]
    logs = [[best_element[1], best_element[3], best_element[2]]]
    if best_element[2] == 1:
      print("All demand are served - stopping search")
      print(f"Cost is {best_element[1]}.")
      return best_element, logs
    print(f"Starting best is {best_element}")
    # iterating search
    for it in range(n_iter):
        cont, new_inds, tabu = move_partition_heuri(inds, possible_set, tabu)
        if not (cont):
          break
        inds = new_inds.copy()
        R = create_demand(possible_set, inds, common_data['fly_time'], verbose=False)
        q = runInParallel(samples, MH_probe, R, common_data)
        for i in range(len(samples)):
            elements.append(q.get())
        print(f"Got all results in {round(time.time() - start, 2)} seconds.")
        print(f"New tabu {tabu}")
        best_cand = sorted(elements, key=lambda x: -x[2])[0]
        print(f"Current cand {best_cand}")
        logs += [[best_cand[1], best_cand[3], best_cand[2]]]
        if best_cand[2] > best_element[2]:
          best_element = best_cand
          print(f"Improved! at {it}")

    return best_element, logs
