
#imports, pulp will be used to solve linear programs.
import numpy as np
import pulp as plp
import itertools
import operator
import matplotlib.pyplot as plt
import networkx as nx
import time
import utils as U
import matplotlib.image as mpimg
import heapq
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import bisect
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import numba
from numba import njit, jit


class schedule_ilp():
  def __init__(
                self,
                helicopters: List[str],
                nodes: List[Tuple[str, int]],
                r: Dict[Tuple[str, str, int], List[Tuple[str, int]]],
                A: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                locations: List[str],
                T: List[int],
                carac_heli: Dict[str, Union[int, str]],
                sinks: Dict[str, Tuple[str, int]],
                parking_fee: Dict[str, int],
                A_w: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                A_s: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                A_f: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                A_g: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                beta: int,
                fees: Dict[str, int],
                fly_time: Dict[Tuple[str, str], int],
                min_takeoff: int,
                refuel_prices: Dict[str, int]) -> None:
    """ first formulation MILP to solve routing problem.
        See overleaf for details.
      --------------
      Args :
              helicopters : list, id of helicopters available to use
              nodes : list, contains all timed nodes
              r : list, contains all requests
              A : list, contains all arcs, waiting, service and deadhead
              locations : list, id of different heliports
              T : list, contains all timesteps (assumed to be separated by one minute)
              carac_heli : dict, contains caracteristics of helicopters
              sinks : dict, maps each helicopter to a unique sink node.
              parking_fee : dict
              A_s : list, service arcs
              A_w : list, waiting arcs
              A_f : list, deadhead arcs
              beta : float, gain from serving a request in dollars - This will become a dict
              to give different values to different requests.
              fees : dict, gives landing fee for each location
              fly_time : dict, give fly time in minutes (rounded) between locations.


      --------------
      Returns :
              None

    """
    self.helicopters = helicopters
    self.A = A
    self.nodes = nodes
    self.r = r
    self.end = U.connect_sink([sinks[h] for h in helicopters], locations, T)
    self.locations = locations
    self.T = T
    self.carac_heli = carac_heli
    self.sinks = sinks
    self.Ah = list(itertools.product(A + self.end, helicopters))
    self.costs = U.create_cost(A_w, A_s, A_f, A_g, self.end, carac_heli, beta, fees, helicopters, fly_time, refuel_prices)
    self.parking_fee = parking_fee
    self.A_g = A_g
    self.fly_time = fly_time
    self.min_takeoff = min_takeoff
    self.A_w = A_w
    self.A_s = A_s
    self.A_f = A_f




  def build_model(self) -> None:
    """ Build the Integers Linear Program instance.
        Instantiate the objective function and the constraint as described in the overleaf.
      --------------
      Args :
              None
      --------------
      Returns :
              None

    """
    # First creates the master problem variables of whether to fly an arc for each helico
    self.arcs = plp.LpVariable.dicts("hFlyArc", (self.helicopters, self.A + self.end), 0, 1, plp.LpInteger)

    #Create auxiliary variable to monitor other constraint, i.e. parking time (and later fuel)
    self.alpha = plp.LpVariable.dicts("alpha", (self.helicopters, self.locations, self.T), 0, None, plp.LpInteger)
    self.delta = plp.LpVariable.dicts("delta", (self.helicopters, self.locations, self.T), 0, None, plp.LpInteger)
    self.park = plp.LpVariable.dicts("park", (self.helicopters, self.locations, self.T), 0, 1, plp.LpInteger)
    self.park_fee = plp.LpVariable.dicts("park_fee", (self.helicopters, self.locations, self.T), 0, 1, plp.LpInteger)
    self.v = plp.LpVariable.dicts("FuelLvl", (self.helicopters, self.locations, self.T), 0, 1100, plp.LpContinuous) #this is the fuel level variable

    #Instantiate the model
    self.model = plp.LpProblem(name="ARP_optim", sense=plp.LpMinimize)

    #Objective function
    self.model += plp.lpSum([self.arcs[h][a]*self.costs[(a, h)] for (a, h) in self.Ah]) + plp.lpSum([self.park_fee[h][l][t] * self.parking_fee[l] for h in self.helicopters for l in self.locations for t in self.T[1:]]), "Total Costs"

    #Adding constraints
    for n in self.nodes:
        for h in self.helicopters:
          if n != (self.carac_heli[h]["start"], 0):
            self.model += plp.lpSum([self.arcs[h][a] for (a, h) in U.inbound(n, self.arcs, [h])]) == plp.lpSum([self.arcs[h][a] for (a, h) in U.outbound(n, self.arcs, [h])])
          self.model += plp.lpSum([self.arcs[h][a] for (a, h) in U.outbound(n, self.arcs, [h])]) <= 1


    for h in self.helicopters:
      self.model += plp.lpSum([self.arcs[h][a] for (a, h) in U.outbound([(self.carac_heli[h]["start"], 0)], self.arcs, [h])]) <= 1
      self.model += plp.lpSum([self.arcs[h][a] for (a, h) in U.inbound([(self.sinks[h], 210)], self.arcs, [h])]) <= 1
      self.model += plp.lpSum([self.arcs[h][a] for (a, h) in U.inbound([(self.sinks[h], 210)], self.arcs, [h])]) <= plp.lpSum([self.arcs[h][a] for (a, h) in U.outbound([(self.carac_heli[h]["start"], 0)], self.arcs, [h])])

    for rs in self.r:
      A_sr = self.r[rs]
      #changing this to enforce Arrive at earliest / Leave at latest constraint (for robustness)
      first = U.inbound(A_sr[0][0], self.arcs, self.helicopters)
      last = A_sr[-1]
      self.model += plp.lpSum([self.arcs[h][a] for (a, h) in first]) == 1
      self.model += plp.lpSum([self.arcs[h][last] for h in self.helicopters]) == 1

    for h in self.helicopters:
      for l in self.locations:
        for t in self.T:
          if t == 0:
            self.model += self.alpha[h][self.carac_heli[h]["start"]][0] == 1
            if l != self.carac_heli[h]["start"]:
              self.model += self.alpha[h][l][0] == 0
            self.model += self.delta[h][l][0] == self.alpha[h][l][0]
            self.model += self.park_fee[h][l][t] == 0

          elif t >= 1:
            self.model += plp.lpSum([self.arcs[h][a] for (a, h) in U.inbound((l, t), self.arcs, [h])]) == self.alpha[h][l][t]
            self.model += self.delta[h][l][t] <= 220 * self.alpha[h][l][t]
            self.model += self.delta[h][l][t] <= self.delta[h][l][t-1] + 1
            self.model += self.delta[h][l][t] >= self.delta[h][l][t - 1] + 1 - 220 * (1 - self.alpha[h][l][t])
            self.model += self.park_fee[h][l][t] >= self.park[h][l][t - 1] - self.park[h][l][t]

          self.model += self.delta[h][l][t] - 15 <= (220 - 15) * (1 - self.park[h][l][t])
          self.model += 15 - self.delta[h][l][t] <= 15 * self.park[h][l][t]

    for h in self.helicopters:
      self.model += self.v[h][self.carac_heli[h]["start"]][0] == self.carac_heli[h]["init_fuel"]

      z = self.carac_heli[h]["conso_per_minute"]
      fuel_cap = self.carac_heli[h]["fuel_cap"]
      for a in self.A_f + self.A_s + self.A_w:
        (i, t1), (j, t2) = a

        if not(a in self.A_w) and not(a in self.end):
            self.model += self.arcs[h][a] * self.min_takeoff <= self.v[h][i][t1]

        self.model += self.v[h][j][t2] <= self.v[h][i][t1] - self.arcs[h][a] * z * self.fly_time[(i, j)] + (1 - self.arcs[h][a]) * fuel_cap
        self.model += self.v[h][j][t2] >= self.v[h][i][t1] - self.arcs[h][a] * z * self.fly_time[(i, j)] - (1 - self.arcs[h][a]) * fuel_cap


      for a in self.A_g:
        (i, t1), (j, t2) = a
        self.model += self.v[h][j][t2] - fuel_cap <= fuel_cap * (1 - self.arcs[h][a])
        self.model += fuel_cap - self.v[h][j][t2] <= fuel_cap * (1 - self.arcs[h][a])
        self.model += self.v[h][i][t1] - self.carac_heli[h]["theta"] <= fuel_cap * (1 - self.arcs[h][a])

      for i in self.locations:
        for t in self.T:
          self.model += self.v[h][i][t] <= fuel_cap


    print("Model Built")
    print("----------")

  def solve(self, max_time: int, opt_gap: float, verbose: bool = 1) -> str:
    """ Solve the Integers Linear Program instance built in self.build_model().
      --------------
      Args :
              max_time : int, maximum running time required in seconds.
              opt_gap : float, in (0, 1), if max_time is None, then the objective value
              of the solution is guaranteed to be at most opt_gap % larger than the true
              optimum.
              verbose : 1 to print log of resolution. 0 for nothing.
      --------------
      Returns :
              Status of the model : Infeasible or Optimal.
              Infeasible indicates that all constraints could not be met.
              Optimal indicates that the model has been solved optimally.

    """
    start = time.time()
    self.model.solve(plp.PULP_CBC_CMD(maxSeconds = max_time, fracGap = opt_gap, msg = verbose, threads=4, mip_start=False, options=["randomCbcSeed 31"]))
    #Get Status
    print("Status:", plp.LpStatus[self.model.status])
    print("Total Costs = ", plp.value(self.model.objective))
    print("Solving time : ", round(time.time() - start, 3), " seconds.")
    return plp.LpStatus[self.model.status]

class META_HEURISTIC():
    def __init__(
                self,
                r: Dict[Tuple[str, str, int], List[Tuple[Tuple[str, Union[np.int64, int]], Tuple[str, Union[np.int64, int]]]]],
                helicopters: List[str],
                carac_heli: Dict[str, Union[int, str]],
                refuel_time: Dict[str, int],
                refuel_price: Dict[str, int],
                locations: List[str],
                nodes: List[Tuple[str, int]],
                parking_fee: Dict[str, int],
                landing_fees: Dict[str, int],
                fly_time: Dict[Tuple[str, int], int],
                T: List[int],
                beta: int,
                mintakeoff: int,
                pen_fuel: int,
                A_s: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                A: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                A_g: List[Tuple[Tuple[str, int], Tuple[str, int]]]) -> None:
        """ Object for the Metaheuristic algorithm. This class contains necesarry function to instantiate the optim problem and to solve it.
            Contains also neighbourhoods operators.


            --------------
            Args :
                    r : dictionnary containing requested flights
                    helicopters : list containing id of helicopters to use
                    carac_heli : dict, contains helicopters' characteristics
                    refuel_time : dict, contains refuel times in minutes for each skyports
                    refuel_price: dict, contains refuel price in dollars for each skyports
                    locations : list, contains all skyports
                    nodes : list, contains all timed node of the TEN
                    parking_fees : dict, contains parking fee in $ for each skyport
                    landing_fees : dict, contains landing fee in $ for each skyport
                    fly_time : dict, gives the fly time duration between any two skyports
                    T : list, contains all timed node of the TEN
                    beta : int, give the gain in serving a demand in $. Constant for now, could be demand-dependent in the future.
                    mintakeoff: int, minimum fuel quantity allowed at takeoff
                    pen_fuel: int, penalisation term for violating the fuel constraint - useful in optim.
                    A_s : list, contains all service arcs
                    A: list, contains all arcs
                    A_g : list, contains all refueling arcs
        """
        self.r = r
        self.helicopters = helicopters
        self.carac_heli = carac_heli
        self.refuel_time = refuel_time
        self.refuel_price = refuel_price
        self.locations = locations
        self.nodes = nodes
        self.parking_fee = parking_fee
        self.landing_fee = landing_fees
        self.fly_time = fly_time
        self.T = T
        self.beta = beta
        self.mintakeoff = mintakeoff

        self.pen_fuel = pen_fuel
        self.service_heli = {h: [] for h in self.helicopters}
        self.request_id = {}
        self.nb_evaluation = 0
        i = 0
        for req in self.r:
          self.request_id[i] = req
          i += 1

        self.mask = np.array([False, True] * len(self.r))
        self.move = 0
        self.A_s = A_s
        self.A_g = A_g
        self.A = A
        self.h_idx_max, self.h_idx_min = self.get_table_indices(len(helicopters), 2 * len(r))
        self.memo_cost_nb, self.cost_rf_nb, self.chain_rf_nb, self.chain_nb, self.req_origin, self.req_dest, self.fly_time_nb = U.numba_dict(r, fly_time, helicopters, carac_heli, refuel_time, locations, refuel_price,
                  parking_fee, landing_fees, beta)

        self.memo_cost = {h:{} for h in helicopters}

    def init_encoding(self) -> None:
        """ Initialize the encoding of solution and store the meaning of each bit : a solution is encoded as an array of lenght 2 * len(self.r) * len(self.helicopters)
            and each bit correspond to a service / refuel operation for a single helicopter.
            Creates the following attributes:
                - indices : dict, gives the meaning of each bit : either a service or a refuel.
                - reverse_indices : dict, reverse operation of indices per helicopter.
                - empty_sol: array, encode an empty solution : all zeros
                - assign: dict, for each heli gives the indices of the bits associated to it
                - ind_type: dict, gives the type of each bit : refuel or service without repetition
                - ind_type_heli: dict, same as previous but for each heli
                - ind_type_seq: dict, gives the type of each bit : refuel or service with repetition
                - queue_swap: numba list, empty will serve to store candidate and costs
                - queue_rm_rf: numba list, empty will serve to store candidate and costs
                - queue_shift: list, empty will serve to store candidate and costs
                - connect_memo: dict, used to memoize certain
                - H: int, number of helicopters
                - table_carac_heli: np.ndarray, PART OF REFACTORING, contains same info as self.carac_heli but under table format
                - heli_start: numba list, contains starting points for helicopters.
            --------------
            Args :
                    None
            --------------
            Returns :
                    None

        """
        indices = {}
        assign = {h:[] for h in self.helicopters}
        demands = list(self.r.keys())
        events = []
        for i in range(2 * len(self.r)):
            if i % 2 == 0:
                x = i // 2
                events.append("Refuel%s"%x)
            else:
                events.append(demands[i // 2])
        events = events * len(self.helicopters)
        ind_type = {"Refuel": [], "Request": []}
        ind_type_seq = {"Refuel": [], "Request": []}
        ind_type_heli = {h:{"Refuel" :np.array([]).astype(int), "Request" :np.array([]).astype(int)} for h in self.helicopters}
        n = 2 * len(self.r)
        for i in range(len(events)):
            indices[i] = events[i]
            if "Ref" in events[i]:
                if not(i%n in ind_type["Refuel"]):
                    ind_type["Refuel"].append(i%n)
                ind_type_seq["Refuel"].append(i)
                ind_type_heli[self.helicopters[i//(2*len(self.r))]]["Refuel"] = np.append(ind_type_heli[self.helicopters[i//(2*len(self.r))]]["Refuel"], i)
            else:
                if not(i%n in ind_type["Request"]):
                    ind_type["Request"].append(i%n)
                ind_type_seq["Request"].append(i)
                ind_type_heli[self.helicopters[i//(2*len(self.r))]]["Request"] = np.append(ind_type_heli[self.helicopters[i//(2*len(self.r))]]["Request"], i)
            assign[self.helicopters[i//(2*len(self.r))]].append(i)

        reverse_indices = {h:{} for h in self.helicopters}
        for k, v in indices.items():
            h = k//(2*len(self.r))
            reverse_indices[self.helicopters[h]][v] = k

        self.reverse_indices = reverse_indices
        self.empty_sol = np.array([0 for j in range(len(events))], dtype='b') #using dtype boolean to reduce memory usage. Reduction by a factor of 8 if default type was used.
        self.indices = indices
        self.assign = assign
        self.ind_type = ind_type
        self.ind_type_heli = ind_type_heli
        self.ind_type_seq = ind_type_seq
        #---
        self.rf_cons = [None] * 5
        self.rf_ev = [None] * 3
        self.len_encod = 2 * len(self.r)
        self.H = len(self.helicopters)
        self.eco = 0
        self.queue_swap = numba.typed.List([(5., np.array([1, 0], dtype='b'))])
        self.queue_rm_rf = numba.typed.List([(5., np.array([1, 0], dtype='b'))])

        self.queue_shift = []
        # ----
        self.heli_start = [None]*self.H
        for h_idx, h in enumerate(self.helicopters):
            self.heli_start[h_idx] = self.carac_heli[h]['start']
        self.heli_start = numba.typed.List(self.heli_start)


        self.table_carac_heli = np.zeros((self.H, 6))
        for h_idx, h in enumerate(self.helicopters):
            for f_idx, f in enumerate(['cost_per_min',
                      'security_ratio',
                      'conso_per_minute',
                      'fuel_cap',
                      'init_fuel',
                      'theta']):
                self.table_carac_heli[h_idx][f_idx] = self.carac_heli[h][f]



    def init_compatibility(self) -> None:
        """ Initializes compatibility between service : i.e. if possible to serve one request after the other.
            Same is evaluated with a refuel inbetween.
            Creates attribute:
                time_compatible: contains all pairs of requests that can be (i.e. there is enough time) served one after the other
                refuel_compatilble: contains all pairs of requests that can be (i.e. there is enough time) served one after the other
                with a refuel after the first one
            --------------
            Args :
                    None
            --------------
            Returns :
                    None

        """

        self.time_compatible, self.refuel_compatible = U.preprocessing(self.r,
                                                                       self.fly_time,
                                                                       self.helicopters,
                                                                       self.carac_heli,
                                                                       self.refuel_time,
                                                                       self.locations,
                                                                       self.H,
                                                                       self.len_encod,
                                                                       self.reverse_indices)



    def init_request_cost(self) -> None:
        """ Compute cost of serving a request.
            Compute a penalty cost such that it is never profitable to not serve a request
            Create attributes :
                - service_cost : dict[request, cost], contains cost for each request
                - pen_unserved : int, penalty for not serving a request to use in optim
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        r_cost = {h:{} for h in self.helicopters}
        for req in self.r:
            for h in self.helicopters:
                c = self.fly_time[(req[0], req[1])] * self.carac_heli[h]["cost_per_min"] + self.landing_fee[req[1]] - self.beta
                r_cost[h][req] = c

        self.service_cost = r_cost
        #determine a good constant for the service penalty
        max_leg = max(self.service_cost[self.helicopters[0]].items(), key=operator.itemgetter(1))[0]
        max_landing_fee = max(self.landing_fee.items(), key=operator.itemgetter(1))[1]
        max_parking_fee = max(self.parking_fee.items(), key=operator.itemgetter(1))[1]
        max_refuel_price = max(self.refuel_price.items(), key=operator.itemgetter(1))[1]
        max_flying_time = max(self.fly_time.items(), key=operator.itemgetter(1))[1]
        c = 0
        for h in self.helicopters:
            c = max(c, self.carac_heli[h]["cost_per_min"])
        self.ref_cost = c * max_flying_time + max_landing_fee + max_parking_fee + max_refuel_price + r_cost[self.helicopters[0]][max_leg]
        self.pen_unserved = self.ref_cost * 3


    @staticmethod
    @njit(cache=True)
    def feasible_fast(arr, len_encod, H,
            time_compatible, refuel_compatible, mask) -> bool:
        """
            DUPLICATE OF FEASIBLE : but this one is jitted

        Test if instance is feasible or not : check for logical infeasibilies but not for
        fuel violations. Fuel violations will be penalized in local search.
        This function is expensive and should be called only when necessary.
        This function will test for :
            - Two different helicopters serving the same demand
            - Refuelling happening for nothing. Refuel slot can be taken in two cases :
            previous demand slot in 1 or the refuel slot correspond to the beginning of service, in which case
            at least 1 demand should be served.
            - Consecutive served demand are time-compatible
            - Consectuvie served demand are time compatible and refuelling is possible inbetween is refuel slot is taken.

            --------------
            Args :
                    arr : np array, encoding of a solution
                    len_encod: int, length of encoding
                    H: int, number of helicopers
                    time_compatible: numba dict, contains pair of request feasible to chain
                    refuel_compatible: numba dict, contains pairs of requests that are feasible to chain with refuel inbetween
                    mask: list, masking for requests bits.
            --------------
            Returns :
                    bool: True iif arr is feasible
        """


        #check for serving conflict
        tab = arr.reshape((H, len_encod))

        if not (np.all(np.sum(tab[:, mask], axis=0) <= 1)):
            return False

        #check for unauthorized refuel or useless ones

        if not (np.all(tab[:, ~mask][:, 0] <= np.sum(tab[:, mask], axis=1))):

            return False
        if not (np.all(tab[:, ~mask][:, 1:] <= tab[:, mask][:, :-1])):

            return False

        for h_i in range(H):

            h_id = - (h_i + 1)
            path = tab[h_i, :]


            if not(np.any(path)):
                continue

            #check for time-compatibility violations and refuel-time violations
            idx = np.where(path == 1)[0]
            if idx[-1] % 2 == 0:

                return False

            if not( idx[0] % 2 == 0) and not((h_id, idx[0] + h_i * len_encod)) in time_compatible:


                return False
            elif idx[0] % 2 == 0 and not((h_id, idx[1] + h_i * len_encod) in refuel_compatible):


                return False

            for i in range(1, len(idx)):
                if idx[i-1] % 2 == 0:
                    continue
                prev, succ = idx[i-1], idx[i]

                if prev % 2 == 1 and succ % 2 == 1 and not((prev + h_i * len_encod, succ + h_i * len_encod) in time_compatible):

                    return False
                if prev % 2 == 1 and succ % 2 == 0 and not((prev + h_i * len_encod, idx[i+1] + h_i * len_encod) in refuel_compatible):

                    return False

        return True

    def feasible(self, arr: np.ndarray) -> bool:
        """ Test if instance is feasible or not : check for logical infeasibilies but not for
        fuel violations. Fuel violations will be penalized in local search.
        This function is expensive and should be called only when necessary.
        This function will test for :
            - Two different helicopters serving the same demand
            - Refuelling happening for nothing. Refuel slot can be taken in two cases :
            previous demand slot in 1 or the refuel slot correspond to the beginning of service, in which case
            at least 1 demand should be served.
            - Consecutive served demand are time-compatible
            - Consectuvie served demand are time compatible and refuelling is possible inbetween is refuel slot is taken.

            --------------
            Args :
                    arr : np array, encoding of a solution
            --------------
            Returns :
                    None
        """

        #check for serving conflict
        tab = arr.reshape((len(self.helicopters), 2 * len(self.r)))

        if not (np.all(np.sum(tab[:, self.mask], axis=0) <= 1)):
            return False

        #check for unauthorized refuel or useless ones

        if not (np.all(tab[:, ~self.mask][:, 0] <= np.sum(tab[:, self.mask], axis=1))):

            return False
        if not (np.all(tab[:, ~self.mask][:, 1:] <= tab[:, self.mask][:, :-1])):

            return False

        for h in self.helicopters:

            h_i = self.helicopters.index(h)
            h_id = - int(h.replace('h', ''))
            path = tab[self.helicopters.index(h), :]

            start = self.carac_heli[h]["start"]
            if not(np.any(path)):
                continue

            #check for time-compatibility violations and refuel-time violations
            idx = np.where(path == 1)[0]
            if idx[-1] % 2 == 0:

                return False
            if not( idx[0] % 2 == 0) and not((h_id, idx[0] + h_i * self.len_encod)) in self.time_compatible:


                return False

            elif idx[0] % 2 == 0 and not((h_id, idx[1] + h_i * self.len_encod) in self.refuel_compatible):


                return False

            for i in range(1, len(idx)):
                if idx[i-1] % 2 == 0:
                    continue
                prev, succ = idx[i-1], idx[i]

                if prev % 2 == 1 and succ in self.ind_type["Request"] and not((prev + h_i * self.len_encod, succ + h_i * self.len_encod) in self.time_compatible):

                    return False
                if prev % 2 == 1 and succ in self.ind_type["Refuel"] and not((prev + h_i * self.len_encod, idx[i+1] + h_i * self.len_encod) in self.refuel_compatible):

                    return False
        return True


    def evaluate_link(
                    self,
                    con_points: List[Tuple[str, int]],
                    h: str,
                    entry_fuel: float,
                    getfuel: bool = False) -> Union[int, Tuple[float, float]]:
        """ *** DEPRECIATED : Should not be used anymore ***
            Compute the cost of a sub-route represented by connection points.
            --------------
            Params :
                    con_points : list, contains connection point of sub-route
                    h: str, id of helicopter involded in sub route
                    entry_fuel: int, level of fuel of h at the beginning of sub route
                    getfuel: bool, whether to return fuel level of h at the end of sub route
            --------------
            Returns :
                    c : int, cost of connection
                    entry_fuel: int, level of fuel at end, if logfuel = True

        """
        c = 0
        if not(con_points):
            if getfuel:
                return c, entry_fuel
            return c
        fuel_level = entry_fuel
        for current, nxt in zip(con_points, con_points[1:]):
            if current[0] == nxt[0]:
                c += self.parking_fee[current[0]] * (nxt[1] - current[1] > 15)
            else:
                if fuel_level <= self.mintakeoff:
                    #c += self.pen_fuel
                    c += self.pen_fuel * (self.mintakeoff - fuel_level)
                c += self.fly_time[(current[0], nxt[0])] * self.carac_heli[h]["cost_per_min"] + self.landing_fee[nxt[0]]
                fuel_level -= self.fly_time[(current[0], nxt[0])] * self.carac_heli[h]["conso_per_minute"]
        if getfuel:
            return c, fuel_level
        return c


    def connects(
                self,
                n: Tuple[str, int],
                m: Tuple[str, int],
                h: str,
                entry_fuel: float) -> Tuple[List[Tuple[str, int]], List[Union[float, int]], float]:
        """ Connects optimally two timed node using aircraft h with entry fuel entry_fuel.
            It gives the cost and the path h will take between timed node n and m.
            --------------
            Args :
                    n: tuple, timed node
                    m: tuple, timed node
                    h: str, id of aircraft making connection
                    entry_fuel: float, fuel level of h when at n.
            --------------
            Returns :
                    - path from n to m, with only breakpoints
                    - [cost of connection, quantity of fuel missing to make the connection]
                    - fuel level at the end of connection

        """


        i, t1 = n
        j, t2 = m
        if i == j:
            straight_path = [(i, t1), (i, t2)]
            c_straight = (t2 - t1 > 15) * self.parking_fee[i]
            return straight_path, [c_straight, 0], entry_fuel

        else:
            free_delta = t2 - t1 - self.fly_time[(i, j)]
            if free_delta < 0:
                return [], [np.inf, 0], entry_fuel
            else:
                if free_delta <= 15 * 2:
                    ta = t1 + int(free_delta/2)
                else:
                    ta = t1 + free_delta * (self.parking_fee[i] < self.parking_fee[j]) + 1 * (self.parking_fee[j] <= self.parking_fee[i])

                link = [(i, t1), (i, ta), (j, ta + self.fly_time[(i, j)]), (j, t2)]
                p_fee = (free_delta > 15 * 2) * min(self.parking_fee[i], self.parking_fee[j])
                fuel_viol = max(0, self.mintakeoff - entry_fuel)
                c = self.fly_time[(i, j)] * self.carac_heli[h]["cost_per_min"] + self.landing_fee[j]
                c += p_fee
                newfuel = entry_fuel - self.fly_time[(i, j)] * self.carac_heli[h]["conso_per_minute"]
                #self.connect_memo[(n, m, h, entry_fuel)] = (link, [c, fuel_viol], newfuel)
                return link, [c, fuel_viol], newfuel


    def refuel_slot(
                    self,
                    n: Tuple[str, int],
                    m: Tuple[str, int],
                    h: str,
                    entry_fuel: float) -> Tuple[List[Tuple[str, int]], float, float]:
        """ Connect n to m with a refuel just after n. Gives cost of the connection.
            --------------
            Args :
                    n: tuple, timed node
                    m: tuple, timed node
                    h: str, id of aircraft making connection
                    entry_fuel: int, fuel level of h when at n.
            --------------
            Returns :
                    rf_cons: list, path of h from n to m - only break points
                    cost: int, cost of the connection
                    entry_fuel: int, fuel level of h at m

        """
        i, t1 = n
        j, t2 = m
        link_path, cost, newfuel = self.connects((i, t1 + self.refuel_time[i]), (j, t2), h, self.carac_heli[h]["fuel_cap"])
        cost[0] += self.refuel_price[i]
        rf_cons = [(i, t1), ('Ref', i, t1)] + link_path
        return rf_cons, cost, newfuel

    def get_indices_heli(self, h_idx: int, len_encod):
        min_idx, max_idx = len_encod * h_idx, len_encod * (h_idx + 1) - 1
        return min_idx, max_idx

    def get_table_indices(self, n_helicopters, len_encod):
        table_min = np.zeros(n_helicopters)
        table_max = np.zeros(n_helicopters)
        for i in range(n_helicopters):
            min_idx, max_idx = self.get_indices_heli(i, len_encod)
            table_min[i] = min_idx
            table_max[i] = max_idx
        return table_max, table_min

    def trace_cost_heli(self, sol, h_idx, start_h, init_fuel_h,
                        conso_h, fuel_cap_h, memo_cost_nb, cost_rf_nb,
                        chain_rf_nb, chain_nb, req_origin, req_dest, fly_time_nb,
                        h_idx_min, h_idx_max, len_encod, mintakeoff):
        min_x, max_x = int(h_idx_min[h_idx]), int(h_idx_max[h_idx])

        hs = sol[min_x:max_x+1].tostring()

        if hs in self.memo_cost[self.helicopters[h_idx]]:
            self.eco += 1

            return self.memo_cost[self.helicopters[h_idx]][hs]
        else:
            res = self.trace_cost_heli_jit(sol, h_idx, start_h, init_fuel_h,
                        conso_h, fuel_cap_h, memo_cost_nb, cost_rf_nb,
                        chain_rf_nb, chain_nb, req_origin, req_dest, fly_time_nb,
                        h_idx_min, h_idx_max, len_encod, mintakeoff)


            self.memo_cost[self.helicopters[h_idx]][hs] = res
            return res

    @staticmethod
    @njit(cache=True)
    def trace_cost_heli_jit(sol, h_idx: int, start_h, init_fuel_h,
                        conso_h, fuel_cap_h, memo_cost_nb, cost_rf_nb,
                        chain_rf_nb, chain_nb, req_origin, req_dest, fly_time_nb,
                        h_idx_min, h_idx_max, len_encod, mintakeoff):
        """
            This one traces the cost only

            start_h, init_fuel_h, conso_h, fuel_cap_h : fixed when h is fixed.
        """


        cost = 0
        current_start, current_dest = start_h, start_h
        current_req_idx = -(h_idx + 1)
        rf = False
        fuel = init_fuel_h
        viol = 0

        for i in range(int(h_idx_min[h_idx]), int(h_idx_max[h_idx]) + 1):
            req_idx_bis = (i - int(h_idx_min[h_idx]))
            if req_idx_bis == 0:
                req_idx = -(h_idx + 1)
            elif req_idx_bis % 2 == 0:
                req_idx = int(req_idx_bis / 2)
            else:
                req_idx = int((req_idx_bis + 1) / 2)

            is_refuel = i % 2 == 0
            if sol[i] == 1:

                if is_refuel:
                    cost += cost_rf_nb[(h_idx, req_idx)]
                    fuel = fuel_cap_h
                    rf = True
                else:
                    if rf:
                        cost += chain_rf_nb[(h_idx, current_req_idx, req_idx)]
                    else:

                        cost += chain_nb[(h_idx, current_req_idx, req_idx)]

                    viol += max(0, mintakeoff - fuel) if current_dest != req_origin[req_idx] else 0
                    fuel -= conso_h * fly_time_nb[(current_dest, req_origin[req_idx])]

                    viol += max(0, mintakeoff - fuel)
                    fuel -= conso_h * fly_time_nb[(req_origin[req_idx], req_dest[req_idx])]
                    current_start, current_dest = req_origin[req_idx], req_dest[req_idx]
                    current_req_idx = req_idx
                    rf = False


        return [cost, viol]

    def compute_cost_heli(
                          self,
                          arr: np.ndarray,
                          h: str,
                          logfuel: bool = False,
                          log: bool = True) -> Union[List[Union[float, int]], Tuple[List[Union[float, int]], List]]:
        """ Compute the cost caused by aircraft h in solution arr.
            If log is true, function also returns the entire path of h containing break points only.

            --------------
            Args :
                    arr : np.array, encoding of a valid solution
                    h: str, id of aircraft
                    log: bool, whether to return path of h in arr
            --------------
            Returns :
                    - operationnal cost of h in arr
                    - additionnal quantity of fuel needed by h in arr
                    - path of h in arr

        """
        self.nb_evaluation += 1

        arr_heli = arr[self.assign[h]]
        ones_index = [i for i in self.ind_type["Request"] if arr_heli[i] == 1]# and i in self.assign[h] ]
        if not (ones_index):
            if log:
                return [0, 0], []
            else:
                return [0, 0]
        fuel = self.carac_heli[h]["init_fuel"]
        start = self.carac_heli[h]["start"]
        cost = 0
        fuel_viol = 0
        if log:
            logs = []
        #---- connecting first request -----
        ref = (arr_heli[0] == 1)
        succ = ones_index[0]
        req_succ = self.r[self.indices[succ]]
        n = (start, 0)
        m = req_succ[0][0]
        if ref:
          points, c, newfuel = self.refuel_slot(n, m, h, fuel)
        else:
          points, c, newfuel = self.connects(n, m, h, fuel)
        cost += c[0]
        fuel_viol += c[1]

        fuel_viol += max(0, self.mintakeoff - newfuel)
        newfuel -= self.carac_heli[h]["conso_per_minute"] * self.fly_time[(self.indices[succ][0], self.indices[succ][1])]
        cost += self.service_cost[h][self.indices[succ]]

        if log:
            logs += points
        if len(ones_index) == 1:
            if log:
                logs += [req_succ[-1][1]]
                return [cost, fuel_viol], logs
            else:
                return [cost, fuel_viol]
        else:

          for current, nxt in zip(ones_index, ones_index[1:]):
            ref = (arr_heli[current+1] == 1)
            req_succ = self.r[self.indices[nxt]]
            req_prev = self.r[self.indices[current]]
            n = req_prev[-1][1]
            m = req_succ[0][0]

            if ref:
              points, c, newfuel = self.refuel_slot(n, m, h, newfuel)
            else:
              points, c, newfuel = self.connects(n, m, h, newfuel)
            cost += c[0]

            fuel_viol += c[1]
            fuel_viol += max(0, self.mintakeoff - newfuel)

            newfuel -= self.carac_heli[h]["conso_per_minute"] * self.fly_time[(self.indices[nxt][0], self.indices[nxt][1])]
            cost += self.service_cost[h][self.indices[nxt]]

            if log:
                logs += points
          if log:
            logs += [req_succ[-1][1]]
            return [cost, fuel_viol], logs
          else:
            return [cost, fuel_viol]


    def cost_breakdown(
                       self,
                       sol: np.ndarray,
                       paths_sol: Dict[str, List[Tuple[str, int]]],
                       A: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                       A_s: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                       A_g: List[Tuple[Tuple[str, int], Tuple[str, int]]]) -> Dict[str, Union[float, int]]:
        """ Break down the cost of solution sol in deadhead cost, refuel cost,
            and compute a conservative lower bound on number of refuel.
            --------------
            Args :
                    sol: np.array, encoding of a valid solution
                    paths_sol: dict, contains path, made of breakpoints, for each aircraft
                    A: list, contains all arcs of TEN
                    A_s: list, contains all service arcs of TEN
                    A_g: list, contains all refueling arcs of TEN
            --------------
            Returns :
                    Dict:
                        - "nb_ref" : number of refuel in sol
                        - "cost_ref" : cost in $ of refuels in sol
                        - "nb_dh" : number of deadheads in sol
                        - "cost_dh" : cost in $ of deadheads in sol
                        - "min_nb_ref" : lower bound on number of refuel necessary

        """
        res = {}
        # --- define measure we are going to take
        nb_ref = 0
        cost_ref = 0
        nb_dh = 0
        cost_dh = 0
        for h in self.helicopters:
            path = paths_sol[h]

            for i in range(1, len(path)):
                prev, succ = path[i-1], path[i]
                a = (prev, succ)

                if not(a in A):
                    continue
                if prev[0] != succ[0]:
                    if a not in A_s:
                        nb_dh += 1
                        cost_dh += self.carac_heli[h]["cost_per_min"] * self.fly_time[(prev[0], succ[0])] + self.landing_fee[succ[0]]


                else:
                    if a in A_g:
                        nb_ref += 1
                        cost_ref += self.refuel_price[prev[0]]

        #---- Get min amount of fuel needed to serve all demands
        min_carb = 0
        for req in self.r:
            min_carb += self.fly_time[(req[0], req[1])] * self.carac_heli["h1"]["conso_per_minute"]
        #---- init fuel in heli
        init_lvl = np.sum([self.carac_heli[h]["init_fuel"] for h in self.helicopters])
        #---- deduces min number of refuels to be made
        cap = self.carac_heli["h1"]["fuel_cap"]
        min_nb_ref = int((min_carb - init_lvl) / cap)
        return {"nb_ref": nb_ref, "cost_ref": cost_ref, "nb_dh": nb_dh, "cost_dh": cost_dh, "min_nb_ref": min_nb_ref}


    def compute_min_slack(
                          self,
                          sol: np.ndarray) -> int:
        """ Computes the minimum of the slack+ in sol.
            See overleaf for definition of slack+/slack.
            --------------
            Args :
                    sol: np.array, encoding of valid solution
            --------------
            Returns :
                    min_slack: int, minimum number of minutes of slack+

        """
        min_slack = len(self.T)
        ones = list(np.where(sol == 1)[0])
        for prev, succ, nxt in zip(ones, ones[1:], ones[2:]):
            rf = False
            if prev // (2*len(self.r)) != succ // (2*len(self.r)):
                continue
            if "Ref" in self.indices[prev]:
                continue
            if "Ref" in self.indices[succ]:
                rf = True
            if rf:

                slack = self.indices[nxt][2] - self.r[self.indices[prev]][-1][1][1] - self.refuel_time[self.indices[prev][1]]
            else:
                slack = self.indices[succ][2] - self.r[self.indices[prev]][-1][1][1]
            print(f"Slack found of {slack} after {self.indices[prev][0]} to {self.indices[prev][1]} at earliest dep : {'{:02d}:{:02d}'.format(*divmod(420 + self.indices[prev][2], 60))}.")
            if slack < min_slack:

                min_slack = slack
        return min_slack


    def compute_cost(
                     self,
                     seq: np.ndarray,
                     getlog: bool = True
                     ) -> Union[Dict, float]:
        """ Compute $ cost of a solution, breaking down cost between helicopters if required.
            --------------
            Args :
                    seq: np.ndarray, encoding of valid solution.
                    getlog: bool, whether to return detailed cost and log of helicopters of just cost.
            --------------
            Returns :
                    if getlog:
                        - dict : cost for each helicopter in seq
                        - float : penalty for not serving demands
                        - dict : log (path) for each heli
                    else:
                        float, cost of seq.

        """

        feas = self.feasible_fast(seq, self.len_encod, self.H,
                                    self.time_compatible, self.refuel_compatible, self.mask)

        if not(feas):
            if getlog:
                return {h: np.inf for h in self.helicopters}, 0, {}
            else:
                return np.inf
        cost = 0
        if getlog:
            cost_heli = {}
            cache_heli = {}
        for h_idx, h in enumerate(self.helicopters):


            if getlog:
                c, log = self.compute_cost_heli(seq, h)
                cost_heli[h] = c
                cache_heli[h] = log
            else:
                op_cost, fuel_viol = self.trace_cost_heli(seq, h_idx, self.carac_heli[h]['start'], self.carac_heli[h]['init_fuel'],
                                                        self.carac_heli[h]['conso_per_minute'], self.carac_heli[h]['fuel_cap'], self.memo_cost_nb, self.cost_rf_nb,
                                                        self.chain_rf_nb, self.chain_nb, self.req_origin, self.req_dest, self.fly_time_nb,
                                                        self.h_idx_min, self.h_idx_max, self.len_encod, self.mintakeoff)
                cost += op_cost + self.pen_fuel * fuel_viol



        if getlog:
            unserved, _ = self.update_served_status(seq)
            cost_pen = len(unserved) * self.pen_unserved
            return cost_heli, cost_pen, cache_heli
        else:
            return cost


    def read_log(
                 self,
                 log: List,
                 h: str,
                 A_s: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                 A_g: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                 A: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                 verbose: bool = True) -> List:
        """ Reads the log of an helicopter to print successions of actions undertook by helicopter in log.
            Stores and output the path of the helicopter, understandable in the TEN framework.
            --------------
            Args :
                    log: list, contains the log of h as computed in compute_cost_heli()
                    h: str, id of helicopter
                    A_s: list, contains all service arcs
                    A_g: list, contains all refuelling arcs
                    A: list, contains all arcs
                    verbose: bool, whether to print actions of helicopters to stdout
            --------------
            Returns :
                    list, contains path of h which can be represented in a TEN : i.e. succession of timed node.

        """


        if len(log) == 0:
            if verbose:
                print(h, " not involved in solution.")
            return []
        path = []
        for current, nxt in zip(log, log[1:]):
            if current == nxt:
                continue
            if "Ref" == nxt[0]:
                nxt = (nxt[1], nxt[2])
                continue
            if "Ref" == current[0]:

                path += [(current[1], current[2]), nxt]
                continue
            dum = (current[0], current[1] + 4)
            if (dum, nxt) in A_s:

                path += [(current[0], current[1] + t) for t in range(1, 5)]
                path += [nxt]
                continue

            if current[0] == nxt[0]:

                path += [(current[0], current[1] + t) for t in range(int(nxt[1] - current[1]))] + [nxt]
                continue

            if nxt[1] - current[1] == self.fly_time[(current[0], nxt[0])]:

                path += [current, nxt]
                continue
        #handles last service in log
        last = log[-1]
        dum = (last[0], last[1] + 4)
        if (dum, nxt) in A_s:

            path += [(last[0], last[1] + t) for t in range(1, 5)]
            path += [nxt]
        #now read the path
        fuel_level = self.carac_heli[h]["init_fuel"]
        if verbose:
            print(h, " starts with ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), " % of fuel.")
            served = 0
            for i in range(1, len(path)):
                prev, succ = path[i-1], path[i]
                a = (prev, succ)
                if not(a in A):
                    continue
                if prev[0] != succ[0]:
                    #change in location
                    fuel_level -= self.fly_time[(prev[0], succ[0])] * self.carac_heli[h]["conso_per_minute"]
                    if a in A_s:
                        served += 1

                        print(h, " starts service in ", prev[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60)), "and finishes in ", succ[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60)), " - Fuel level at arrival: ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), "%", " - Minimum level reached " * (fuel_level < 325) )
                    else:
                        print(h, " leaves from ", prev[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60)), "and arrives in ", succ[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60)), " - Fuel level at arrival: ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), "%", " - Minimum level reached " * (fuel_level < 325))

                else:
                    if a in A_g:
                        fuel_level = self.carac_heli[h]["fuel_cap"]
                        print(h, " starts refueling in ", prev[0], " at ",'{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60)), ", finishes at ", '{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60)), " - Fuel level at arrival: ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), "%", " - Minimum level reached " * (fuel_level < 325))
            print("")

        return path



    def fuel_checker(
                    self,
                    sol: np.ndarray) -> bool:
        """ Checks whether sol violates the fuel constraints, i.e. if helicopters are flying without enough fuel.
            --------------
            Args :
                    sol: np.ndarray, encoding of a valid solution

            --------------
            Returns :
                    bool: True iif solution DOES NOT violate the fuel constraint.

        """
        ch, _, _ = self.compute_cost(sol)
        fuel_viol = 0
        for h in ch:
            fuel_viol += ch[h][1]
        return fuel_viol == 0


    def apply_neigh(
                    self,
                    n: Callable,
                    seq: np.ndarray,
                    served: List[Tuple[str, str, int]],
                    unserved: List[Tuple[str, str, int]],
                    eps: float,
                    u: float,
                    fail: int,
                    s: int,
                    stag: int,
                    stagr: int,
                    perf_over_time: List[Tuple[float, float]],
                    current_cost: float,
                    best_cost: float,
                    best_sol: np.ndarray,
                    start: float,
                    frozen_bits: List[int] = []) -> Tuple[int, int, np.ndarray, float, np.ndarray, float, int, int, List[Tuple[float, float]]]:
        """ Applies a neighbourhoods operator : i.e. computes all neighbours of seq as defined by operator n and return updated values for current and best points.
            --------------
            Args :
                    n: Callable, neighbourhood operator
                    seq: np.ndarray, encoding of a valid soluton
                    served: list, contains all served requests in seq
                    unserved: list, contains all requests that are not served in seq
                    eps: float, probability of accepting a neighbour with worse cost as new current sol.
                    u: float, between 0 and 1, drawn from random uniform
                    fail: int, current number of failed attempts to use n
                    s: int, current number of use of n
                    stag: int, current number of stagnation steps
                    stagr: int, current number of stagnation steps in MH
                    perf_over_time: list, contains valid solution with improving cost obtained so far
                    current_cost: float, cost of seq
                    best_cost: float, cost of best_sol
                    best_sol: np.ndarray, best (valid) solution obtained so far
                    start: float, starting time of the MH
                    frozen_bits: list, contains all bits in the encoding that are frozen, i.e. optim is not allowed to change them.


            --------------
            Returns :
                    updated value for:
                        - fail
                        - s
                        - seq
                        - current_cost
                        - best_sol
                        - best_cost
                        - stag
                        - stagr
                        - perf_over_time
        """
        s += 1
        if self.jit_neigh:
            cand, cost_cand, self.move = n(seq, served, unserved, frozen_bits=frozen_bits)
        else:
            cand, cost_cand = n(seq, served, unserved, frozen_bits=frozen_bits)
        fail += 1

        if (cost_cand < current_cost or u < eps) and cost_cand < np.inf:
            current_cost = cost_cand
            seq = cand
            fail -= 1

            if current_cost < best_cost and self.fuel_checker(cand):
                stag = 0
                stagr = 0
                self.found_better = True
                best_sol = seq.copy()
                best_cost = current_cost
                #-- adding log for proportion of served demand at this stage
                unserved, served = self.update_served_status(best_sol)
                perf_over_time.append((best_cost, round(time.time() - start, 2), len(served)/len(self.r)))

        return fail, s, seq, current_cost, best_sol, best_cost, stag, stagr, perf_over_time



    def VNS(
            self,
            current_sol: np.ndarray,
            n_iter: int,
            verbose: bool = True,
            eps: float = 0.1,
            no_imp_limit: int = 1700,
            random_restart: int = 900,
            frozen_bits: List = []) -> Tuple[np.ndarray, float, List[float], List[float], List[Tuple[float, float]], List[float], str]:
        """ Metaheuristic main component. Implement a variant of variable neighbourhoods search, the search alternates between using different neighbourhoods operators
            switching whenever no progress is made.
            --------------
            Args :
                    current_sol: np.ndarray, valid encoding of a solution
                    n_iter: int, number of iteration to be made in the algo
                    verbose: bool, whether to print info to stdout while running
                    eps: float, probability of accepting non-improving candidate during the search
                    no_imp_limit: int, number of stagnation iteration after which the algo should stop
                    random_restart: int, number of stagnation iteration after which the algo should restart from random point *** DEPRECIATED ***
                    frozen_bits: list, frozen bits in the solution which the optim is not allowed to change.
            --------------
            Returns :
                    best_sol: np.ndarray, encoding of valid solution - best found during the search
                    best_cost: float, cost of best_sol
                    cache_cost_best: list, best_cost encountered during the search, in order
                    cache_cost_current: list, cost of solution explored during the search, in order
                    perf_over_time: list[Tuple], contains sequence of best cost obtained with time it took : useful for anytime aspect
                    distances: distance of explored solutions to starting point
                    text_log: str, log in markdown format for dashboard use.

        """

        start = time.time()
        self.found_better = False
        self.jit_neigh = False
        text_log = ""
        self.pen_fuel = 0
        #current_sol
        served = [self.indices[i] for i in self.ind_type_seq['Request'] if current_sol[i]==1]
        unserved = list(set(self.r.keys()) - set(served))
        init_sol = current_sol.copy()
        seq = current_sol.copy()
        current_cost = self.compute_cost(seq, getlog=False) + len(unserved) * self.pen_unserved
        best_cost = current_cost
        best_sol = current_sol.copy()
        rf, rp, sw, swr, rm_rf = 0, 0, 0, 0, 0
        cache_cost_best = []
        cache_cost_current = []
        fail_path = 0
        fail_swap = 0
        fail_fuel = 0
        fail_swap_req = 0
        fail_rm_rf = 0
        stag = 0
        stagr = 0
        fac = 1.2
        perf_over_time = []
        distances = []
        for it in range(n_iter):
            self.jit_neigh = False
            distances.append(np.linalg.norm(seq - init_sol, ord=1))
            stag += 1
            stagr += 1

            if stag > no_imp_limit:
                if self.fuel_checker(best_sol) and self.found_better:
                    break

            if (1 + it) % 100 == 0:
                if self.pen_fuel < 100 * self.pen_unserved:
                    self.pen_fuel = max(self.pen_fuel * fac, self.pen_fuel)
                    if self.pen_fuel == 0:
                        self.pen_fuel = 1
                    served_curr = [self.indices[i] for i in self.ind_type_seq['Request'] if seq[i]==1]
                    unserved_curr = list(set(self.r.keys()) - set(served_curr))
                    served_best = [self.indices[i] for i in self.ind_type_seq['Request'] if best_sol[i]==1]
                    unserved_best = list(set(self.r.keys()) - set(served_best))

                    current_cost = self.compute_cost(seq, getlog=False) + len(unserved_curr) * self.pen_unserved
                    current_bestb = self.compute_cost(best_sol, getlog=False) + len(unserved_best) * self.pen_unserved
                    if current_bestb > best_cost:
                        stag = 0
                        stagr = 0
                    best_cost = current_bestb


            if verbose and it%50 == 0:
                print("Iteration ", it)
                print("Current cost is ", current_cost)
                print("Best cost is ", best_cost)
                print("Current penalties are : Fuel :", self.pen_fuel, " | Request : ", self.pen_unserved)
                print("--------")
                print("Path moves :", rp, "| Refuel moves :", rf, " | Swap path moves :", sw, " | Swap request path moves :", swr, " | Remove and Refuel moves :", rm_rf)
            if it%50 == 0:
                text_log += f"  \nIteration {it}  \nCurrent cost is {current_cost}  \nBest cost is {best_cost}  \nCurrent penalties are : Fuel : {self.pen_fuel}  | Request :  {self.pen_unserved}  \n------  \nPath moves : {rp} | Refuel moves : {rf} | Swap path moves : {sw} | Request : {swr}. "
            #First neighbourhood : the demands
            if it > 1:
                cache_cost_best.append(best_cost)
                cache_cost_current.append(current_cost)
            if fail_path >= 2 and fail_fuel >= 2 and fail_swap >= 2 and fail_swap_req >= 2 and fail_rm_rf>=2:
              fail_swap = 0
              fail_path = 0
              fail_fuel = 0
              fail_swap_req = 0
              fail_rm_rf = 0
            #if there are frozen bits we will not use the swap path neighbourhood
            if frozen_bits:
                fail_swap = 10
            u = np.random.random()
            if fail_swap < 2:
                N = self.swap_paths_neigh
                fail_swap, sw, seq, current_cost, best_sol, best_cost, stag, stagr, perf_over_time = self.apply_neigh(N, seq, served, unserved, eps, u, fail_swap, sw, stag, stagr, perf_over_time, current_cost, best_cost, best_sol, start)
                continue
            if fail_path < 2:
                rp += 1

                served = [self.indices[i] for i in self.ind_type_seq['Request'] if seq[i]==1]
                unserved = list(set(self.r.keys()) - set(served))
                N = self.shift_neigh

                fail_path, rp, seq, current_cost, best_sol, best_cost, stag, stagr, perf_over_time = self.apply_neigh(N, seq, served, unserved, eps, u, fail_path, rp, stag, stagr, perf_over_time, current_cost, best_cost, best_sol, start, frozen_bits=frozen_bits)
                continue
            if fail_swap_req < 2:
                swr += 1

                unserved, served = self.update_served_status_fast(seq, len(self.r), self.len_encod)

                N = self.swap_neigh
                self.jit_neigh = True
                fail_swap_req, swr, seq, current_cost, best_sol, best_cost, stag, stagr, perf_over_time = self.apply_neigh(N, seq, served, unserved, eps, u, fail_swap_req, swr, stag, stagr, perf_over_time, current_cost, best_cost, best_sol, start, frozen_bits=frozen_bits)
                continue
            if fail_fuel < 2:
                fail_fuel += 1
                rf += 1
                N = self.refuel_neigh
                fail_fuel, rf, seq, current_cost, best_sol, best_cost, stag, stagr, perf_over_time = self.apply_neigh(N, seq, served, unserved, eps, u, fail_fuel, rf, stag, stagr, perf_over_time, current_cost, best_cost, best_sol, start, frozen_bits=frozen_bits)
                continue
            if fail_rm_rf < 2:
                fail_rm_rf += 1
                rm_rf += 1
                unserved, served = self.update_served_status_fast(seq, len(self.r), self.len_encod)
                N = self.rm_rf_neigh
                self.jit_neigh = True
                fail_rm_rf, rm_rf, seq, current_cost, best_sol, best_cost, stag, stagr, perf_over_time = self.apply_neigh(N, seq, served, unserved, eps, u, fail_rm_rf, rm_rf, stag, stagr, perf_over_time, current_cost, best_cost, best_sol, start, frozen_bits=frozen_bits)
                continue


        if verbose:
            print(f"Meta Heuristic VNS done in : {round(time.time() - start, 2)} seconds. Move per second : {self.move / round(time.time() - start, 2)}")
        text_log += f"  \nMeta Heuristic VNS done in : {round(time.time() - start, 2)} seconds. Move per second : {self.move / round(time.time() - start, 2)}. Iterations {it}"
        return best_sol, best_cost, cache_cost_best, cache_cost_current, perf_over_time, distances, text_log

    def restart_heuri(
                      self,
                      queue: List[np.ndarray]) -> np.ndarray:
        """ Heuristic to restart VNS using a set of solutions.
            --------------
            Args :
                    queue: list of valid solution encoding
            --------------
            Returns :
                    np.ndarray, valid solution encoding

        """
        mat = queue[0].reshape((1, len(queue[0])))
        for k in range(1, len(queue)):
            mat = np.concatenate((mat, queue[k].reshape((1, len(queue[0])))), axis=0)

        return np.product(mat, axis=0)


    def check_refuel_ins(
                        self,
                        seq: np.ndarray,
                        bit: int) -> bool:
        """ Check whether inserting a refuel in seq at position bit results in valid solution,
        assuming seq is valid in the first place.
            --------------
            Args :
                    seq: np.ndarray, valid solution encoding
                    bit: int, bit to try and flip (adding a refuel)
            --------------
            Returns :
                    bool: True iif a refuel can be inserted in spot bit.

        """
        h = bit // (2 * len(self.r))
        h_name = self.helicopters[h]
        h_id = -int(h_name.replace('h',''))
        start = self.carac_heli[self.helicopters[h]]["start"]
        if seq[bit] == 0:
            return True
        if self.indices[bit] == "Refuel0":
            if not(np.any(seq[self.assign[self.helicopters[h]]][1:])):
                return False
            else:
                i = 1
                nxt = None
                while bit + i in self.assign[self.helicopters[h]]:
                    if seq[bit + i] == 1:
                        nxt = bit + i
                        break
                    i += 1

                if not((h_id, nxt) in self.refuel_compatible):

                    return False
        else:
            if seq[bit - 1] == 0:
                return False
            prev = bit - 1
            i = 1
            nxt = None
            while bit + i in self.assign[self.helicopters[h]]:
                if seq[bit + i] == 1:
                    nxt = bit + i
                    break
                i += 1

            if not(nxt) or nxt not in self.assign[self.helicopters[h]]:
                return False

            if not((prev, nxt) in self.refuel_compatible):
                return False
        return True

    @staticmethod
    @njit(cache=True)
    def check_insertion_fast(seq, h, h_name, ind_req, len_encod, time_compatible, refuel_compatible) -> bool:
        """ Checks whether req can be inserted in solution seq in the path of h.
            Part of jittification
            --------------
            Args :
                    seq: np.ndarray, encoding of a valid solution
                    h: int, id of helicopter
                    h_name: str, helicopter id
                    ind_req: int, id of request
                    len_encod: int, length of encoding
                    time_compatible: numba dict, contains all pairs of chainable requests
                    refuel_comptatible: numba dict, contains all pairs of chainable requests with
                                        refuel inbetween.
            --------------
            Returns :
                    bool: true iif req can be inserted in h's path

        """

        i = 1
        h_i = -(h + 1)
        prev = None
        while (ind_req - i) // len_encod == h:
            if (ind_req - i)%2 == 1 and seq[int(ind_req - i)] == 1:
                prev = ind_req - i
                break
            i += 1
        i = 1
        nxt = None
        while (ind_req + i) // len_encod == h:
            if seq[int(ind_req + i)] == 1:
                nxt = ind_req + i
                break
            i += 1

        if nxt is not None and nxt // len_encod == h and not((ind_req, nxt) in time_compatible):
                return False
        if prev is not None and prev // len_encod == h:
            if seq[int(prev + 1)] == 1 and not ((prev, ind_req) in refuel_compatible):
                return False
            if not ((prev, ind_req) in time_compatible):
                return False
        elif prev is None or not(prev // len_encod == h):

            if seq[int(len_encod * h)] == 1 and not ((h_i, ind_req) in refuel_compatible):
                return False
            if not ((h_i, ind_req) in time_compatible):
                return False
        return True

    def check_insertion(
                        self,
                        seq: np.ndarray,
                        h: str,
                        req: Tuple[str, str, int]) -> bool:
        """ Checks whether req can be inserted in solution seq in the path of h.
            --------------
            Args :
                    seq: np.ndarray, encoding of a valid solution
                    h: str, helicopter id
                    req: tuple, request to be inserted
            --------------
            Returns :
                    bool: true iif req can be inserted.

        """
        ind_req = self.reverse_indices[h][req]
        h_i = self.helicopters.index(h)

        start = self.carac_heli[h]["start"]

        i = 1
        prev = None
        while ind_req - i in self.assign[h]:
            if ind_req - i in self.ind_type_heli[h]['Request'] and seq[ind_req - i] == 1:
                prev = ind_req - i
                break
            i += 1
        i = 1
        nxt = None
        while ind_req + i in self.assign[h]:
            if seq[ind_req + i] == 1:
                nxt = ind_req + i
                break
            i += 1


        if nxt and nxt in self.assign[h] and not((ind_req, nxt) in self.time_compatible):
                return False
        if prev and prev in self.assign[h]:
            if seq[prev + 1] == 1 and not ((prev, ind_req) in self.refuel_compatible):
                return False
            if not ((prev, ind_req) in self.time_compatible):
                return False
        elif not(prev) or not(prev in self.assign[h]):

            if seq[self.reverse_indices[h]["Refuel0"]] == 1 and not ((h, ind_req) in self.refuel_compatible):
                return False
            if not ((h, ind_req) in self.time_compatible):
                return False
        return True

    def swap_paths_neigh(
                        self,
                        seq: np.ndarray,
                        served: Any = None,
                        unserved: Any = None,
                        frozen_bits: Any = None) -> Tuple[np.ndarray, float]:
        """ Operator for the swap path neighbourhoods. Compute all neighbours of seq and returns a random one among
            the 3 best.
            A neighbour of seq is a solution in which helicopters exchanges their route without any other change.
            --------------
            Args :
                    seq: np.ndarray, valid encoding of solution
                    served: none
                    unserved: none
                    frozen_bits: none
            --------------
            Returns :
                    - valid solution encoding
                    - cost of solution returned

        """
        queue = []

        k = 3
        _, pen, _ = self.compute_cost(seq)

        for i in range(len(self.helicopters)):
            rolled_seq = np.roll(seq, 2*len(self.r)*i)
            cost = self.compute_cost(rolled_seq, getlog=False)
            cost += pen
            #insertion sort
            if len(queue) < k:
                heapq.heappush(queue, (-cost, i, rolled_seq))
            else:
                _ = heapq.heappushpop(queue, (-cost, i, rolled_seq))

        np.random.shuffle(queue)
        return queue[0][2], -queue[0][0]




    def shift_neigh(
                    self,
                    seq: np.ndarray,
                    served: List[Tuple[str, str, int]],
                    unserved: List[Tuple[str, str, int]],
                    frozen_bits: List = []) -> Tuple[np.ndarray, float]:
        """ Operator for the shift neighbourhoods. Compute all neighbours of seq and returns a random one among
            the 3 best.
            A neighbour of seq is a solution in which one request is reassigned to another
             (different from the one it is assigned to in seq) helicopter or marked as unserved.

            --------------
            Args :
                    seq: np.ndarray, valid encoding of solution
                    served: list, contains served requests in seq
                    unserved: list, contains unserved requests in seq
                    frozen_bits: list, contains frozen bits, not allowed to be touched during the search
            --------------
            Returns :
                    - valid solution encoding
                    - cost of solution returned

        """
        cost_heli, cost_pen, _ = self.compute_cost(seq)
        cost = cost_pen
        for h in cost_heli:
            cost += cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]

        to_freeze = frozen_bits != []
        alld = list(self.r.keys())  #unserved+served
        np.random.shuffle(alld)

        self.queue_shift.clear()
        self.queue_shift.append((-cost, -1, seq))
        k = 3
        i = 0
        for req in alld:
            if req in unserved:
                for h_idx, h in enumerate(self.helicopters):
                    #check for frozen parts
                    if to_freeze:
                        if self.reverse_indices[h][req] in frozen_bits:
                            continue

                    self.move = self.move + 1
                    cand = self.add_request(seq, req, h)

                    feas = self.check_insertion_fast(cand, self.helicopters.index(h),
                                                    h,
                                                    self.reverse_indices[h][req],

                                                    self.len_encod,
                                                    self.time_compatible,
                                                    self.refuel_compatible)

                    if not (feas):
                        continue

                    op_cand, viol_cand = self.trace_cost_heli(cand, h_idx, self.carac_heli[h]['start'], self.carac_heli[h]['init_fuel'],
                                                        self.carac_heli[h]['conso_per_minute'], self.carac_heli[h]['fuel_cap'],
                                                        self.memo_cost_nb, self.cost_rf_nb,
                                                        self.chain_rf_nb, self.chain_nb, self.req_origin, self.req_dest,
                                                        self.fly_time_nb,
                                                        self.h_idx_min, self.h_idx_max, self.len_encod, self.mintakeoff)
                    cand_cost_h = op_cand + self.pen_fuel * viol_cand
                    cost_cand = cost - (cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]) + cand_cost_h - self.pen_unserved

                    if cost_cand < np.inf:
                        if len(self.queue_shift) < k:
                            i = i + 1
                            heapq.heappush(self.queue_shift , (-cost_cand, i, cand))
                        else:
                            i = i + 1
                            _ = heapq.heappushpop(self.queue_shift , (-cost_cand, i, cand))


            else:
                if to_freeze:
                    h_i = [j for j in range(len(self.helicopters)) if seq[self.reverse_indices["h1"][req] + self.len_encod * j] == 1][0]
                    if self.reverse_indices[self.helicopters[h_i]][req] in frozen_bits:
                        continue
                self.move = self.move + 1
                ind_req = self.reverse_indices['h1'][req]
                cand, h_i = self.remove_request_fast(seq, ind_req, self.len_encod, self.H)

                h = self.helicopters[h_i]
                op_cand, viol_cand = self.trace_cost_heli(cand, h_i, self.carac_heli[h]['start'], self.carac_heli[h]['init_fuel'],
                                                        self.carac_heli[h]['conso_per_minute'], self.carac_heli[h]['fuel_cap'],
                                                        self.memo_cost_nb, self.cost_rf_nb,
                                                        self.chain_rf_nb, self.chain_nb, self.req_origin, self.req_dest, self.fly_time_nb,
                                                        self.h_idx_min, self.h_idx_max, self.len_encod, self.mintakeoff)
                cand_cost_h = op_cand + self.pen_fuel * viol_cand


                cost_cand = cost - (cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]) + cand_cost_h + self.pen_unserved

                if cost_cand < np.inf:
                    if len(self.queue_shift) < k:
                        i = i + 1
                        heapq.heappush(self.queue_shift, (-cost_cand, i, cand))
                    else:
                        i = i + 1
                        _ = heapq.heappushpop(self.queue_shift, (-cost_cand, i, cand))


                for id_hb, hb in enumerate(self.helicopters):
                    if to_freeze:
                        if self.reverse_indices[hb][req] in frozen_bits:
                            continue
                    if hb != h:
                        self.move = self.move + 1
                        candu = self.add_request(cand, req, hb)
                        feas = self.check_insertion_fast(candu, id_hb,
                                                    hb,
                                                    self.reverse_indices[hb][req],
                                                    self.len_encod,
                                                    self.time_compatible,
                                                    self.refuel_compatible)
                        if not(feas):
                            continue
                        op_candu, viol_candu = self.trace_cost_heli(candu, id_hb, self.carac_heli[hb]['start'], self.carac_heli[hb]['init_fuel'],
                                                        self.carac_heli[hb]['conso_per_minute'], self.carac_heli[hb]['fuel_cap'], self.memo_cost_nb, self.cost_rf_nb,
                                                        self.chain_rf_nb, self.chain_nb, self.req_origin, self.req_dest, self.fly_time_nb,
                                                        self.h_idx_min, self.h_idx_max, self.len_encod, self.mintakeoff)

                        cand_cost_hb = op_candu + self.pen_fuel * viol_candu
                        cost_candu = cost_cand - (cost_heli[hb][0] + self.pen_fuel * cost_heli[hb][1]) + cand_cost_hb - self.pen_unserved

                        if cost_candu < np.inf:
                            if len(self.queue_shift) < k:
                                i = i + 1
                                heapq.heappush(self.queue_shift, (-cost_candu, i, candu))
                            else:
                                i = i + 1
                                _ = heapq.heappushpop(self.queue_shift, (-cost_candu, i, candu))

        np.random.shuffle(self.queue_shift)
        return self.queue_shift[0][2], -self.queue_shift[0][0]


    def swap_neigh(
                    self,
                    seq,
                    served,
                    unserved=None,
                    frozen_bits=[]):
        """ Wrapper for the swap neighbourhoods.

            --------------
            Args :
                    seq: np.ndarray, valid encoding of solution
                    served: list, contains served requests in seq
                    unserved: list, contains unserved requests in seq
                    frozen_bits: list, contains frozen bits, not allowed to be touched during the search
            --------------
            Returns :
                    - valid solution encoding
                    - cost of solution returned

        """
        #getting cost in numba
        cost_heli, cost_pen, _ = self.compute_cost(seq)
        cost = cost_pen
        for h in cost_heli:
            cost += cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]

        cost_heli_nb = numba.typed.Dict.empty(
                        key_type=numba.typeof('h1'),
                        value_type= numba.typeof(1.),
                        )

        for h in cost_heli:
            cost_heli_nb[h] = float(cost_heli[h][0] + self.pen_fuel * cost_heli[h][1])

        #setting variables for fast swap neigh
        to_freeze = frozen_bits != []
        np.random.shuffle(served)
        id_res = np.random.randint(3)
        self.queue_swap.clear()
        self.queue_swap.append((float(cost), seq))
        return self.swap_neigh_fast(seq,
                                numba.typed.List(served),
                                numba.typed.List([-6]),
                                len(self.r),
                                self.len_encod,
                                self.served_by_fast,
                                self.get_reverse_indice,
                                self.remove_request_fast,
                                numba.typed.List(self.helicopters),
                                self.add_request_jit,
                                self.time_compatible,
                                self.refuel_compatible,
                                self.memo_cost_nb,
                                self.cost_rf_nb,
                                self.chain_rf_nb,
                                self.chain_nb,
                                self.req_origin,
                                self.req_dest,
                                self.fly_time_nb,
                                self.h_idx_min,
                                self.h_idx_max,
                                self.mintakeoff,
                                self.pen_fuel,
                                self.H,
                                self.trace_cost_heli_jit,
                                self.check_insertion_fast,
                                self.queue_swap,
                                cost_heli_nb,
                                self.heli_start,
                                self.table_carac_heli,
                                cost, to_freeze, id_res, self.move, U.sort_ins
                              )

    @staticmethod
    @njit
    def swap_neigh_fast(seq: np.ndarray,
                served,
                frozen_bits,
                len_r,
                len_encod,
                served_by_fast,
                get_reverse_indice,
                remove_request_fast,
                helicopters,
                add_request_jit,
                time_compatible,
                refuel_compatible,
                memo_cost_nb,
                cost_rf_nb,
                chain_rf_nb,
                chain_nb,
                req_origin,
                req_dest,
                fly_time_nb,
                h_idx_min,
                h_idx_max,
                mintakeoff,
                pen_fuel,
                H, trace_cost_heli, check_insertion_fast, queue_swap,
                cost_heli,
                heli_start,
                table_carac_heli, cost, to_freeze, id_res, move, sort_ins
              ):
        """ Operator for the swap path neighbourhoods. Compute all neighbours of seq and returns a random one among
            the 3 best.
            A neighbour of seq is a solution in which 2 requests are swapped : if r1 is served by h1 in seq and r2 by h2, the
            neighbour is defined as the solution in which r1 is served by h2 and r2 is served by h1.
            --------------
            Args :
                    served: list of served requests, contains ints
                    frozen_bits: list of bits that cannot be touched, contains int
                    len_r: int, number of requests in problem instance
                    len_encod: int, length of encoding
                    served_by_fast:  method of self.
                    get_reverse_indice: method of self.
                    remove_request_fast: method of self.
                    helicopters: list of helictoper ids numba typed
                    add_request_jit: method of self.
                    time_compatible: numba dict, contains all requests that can be chained
                    refuel_compatible: numba dict, contains all requests that be chained with refuel inbetween
                    memo_cost_nb: numba dict, contains memoized costs
                    cost_rf_nb: numba dict, contains cost for any heli to start by serving a request with refuel
                    chain_rf_nb: numba dict, contains cost of chaining any pair of requests with refuel inbetween
                    chain_nb: numba dict, contains cost of chaining any pair of requests
                    req_origin: numba dict, contains request origin
                    req_dest: numba dict, contains request destination
                    fly_time_nb: numba dict, contains fly times
                    h_idx_min: int, min index of h in solution encoding
                    h_idx_max: int max index of h in solution encoding
                    mintakeoff: float, minimum amount of fuel required to takeoff
                    pen_fuel: float, fuel penalty term
                    H: int, number of heli
                    trace_cost_heli: method of self.
                    check_insertion_fast: method of self.
                    queue_swap: numba list to use for insertion sort of cands
                    cost_heli: numba dict, contains cost and violation for each heli
                    heli_start: np.ndarray, contains starting locations of each heli
                    table_carac_heli: np.ndarray, contains helicopters caracs
                    cost: float, cost of actual solution
                    to_freeze: bool, true iif frozen_bits is not empty
                    id_res: int between 0 and 2, for random selection of neighbour at the end of function
                    move : int, number of moves made so far
                    sort_ins: function insertion sort
            --------------
            Returns :
                    - valid solution encoding
                    - cost of solution returned

        """


        k_max = 3
        served_by_table = served_by_fast(seq, len_r, len_encod)

        for k in range(len(served)):
            for l in range(k):
                if k == l:
                    continue

                r1 = served[k]
                r2 = served[l]

                if served_by_table[r1-1] == served_by_table[r2-1]:
                    continue
                #-- checking for frozen parts
                if to_freeze:
                    if get_reverse_indice(r1-1, len_encod, served_by_table[int(r1)-1]) in frozen_bits:
                        continue
                    if get_reverse_indice(r2-1, len_encod, served_by_table[int(r2)-1]) in frozen_bits:
                        continue


                #swapping
                move = move + 4
                ind_r1 = get_reverse_indice(r1-1, len_encod, 0)
                ind_r2 = get_reverse_indice(r2-1, len_encod, 0)
                cand, h1_i = remove_request_fast(seq, ind_r1, len_encod, H)
                h1 = helicopters[h1_i]
                cand, h2_i = remove_request_fast(cand, ind_r2, len_encod, H)
                h2 = helicopters[h2_i]
                cand = add_request_jit(cand, r2-1, len_encod, h1_i, get_reverse_indice)
                cand = add_request_jit(cand, r1-1, len_encod, h2_i, get_reverse_indice)

                feas1 = check_insertion_fast(cand, h1_i,
                                                     h1,
                                                     get_reverse_indice(r2-1, len_encod, h1_i),
                                                     len_encod,
                                                     time_compatible,
                                                     refuel_compatible)
                feas2 = check_insertion_fast(cand, h2_i,
                                                     h2,
                                                     get_reverse_indice(r1-1, len_encod, h2_i),
                                                     len_encod,
                                                     time_compatible,
                                                     refuel_compatible)
                feas = feas1 and feas2

                if not(feas):
                    continue
                op_cand1, viol_cand1 = trace_cost_heli(cand, h1_i, heli_start[h1_i], table_carac_heli[h1_i, 4],
                                                        table_carac_heli[h1_i, 2], table_carac_heli[h1_i, 3],
                                                        memo_cost_nb, cost_rf_nb,
                                                        chain_rf_nb, chain_nb, req_origin, req_dest, fly_time_nb,
                                                        h_idx_min, h_idx_max, len_encod, mintakeoff)

                cand1_cost_h = op_cand1 + pen_fuel * viol_cand1
                op_cand2, viol_cand2 = trace_cost_heli(cand, h2_i, heli_start[h2_i], table_carac_heli[h2_i, 4],
                                                        table_carac_heli[h2_i, 2], table_carac_heli[h2_i, 3],
                                                        memo_cost_nb, cost_rf_nb,
                                                        chain_rf_nb, chain_nb, req_origin, req_dest, fly_time_nb,
                                                        h_idx_min, h_idx_max, len_encod, mintakeoff)

                cand2_cost_h = op_cand2 + pen_fuel * viol_cand2

                cost_cand = cost - (cost_heli[h1] + cost_heli[h2]) + cand1_cost_h + cand2_cost_h

                if cost_cand < np.inf:
                    len_q = len(queue_swap)
                    queue_swap = sort_ins(len_q, k_max, queue_swap, cand, cost_cand)



        if id_res > len(queue_swap):
            return queue_swap[-1][1], queue_swap[-1][0], move
        return queue_swap[id_res][1], queue_swap[id_res][0], move


    def refuel_neigh(
                    self,
                    seq: np.ndarray,
                    served: Any = None,
                    unserved: Any = None,
                    frozen_bits: List = []) -> Tuple[np.ndarray, float]:
        """ Operator for the swap path neighbourhoods. Compute all neighbours of seq and returns a random one among
            the 3 best.
            A neighbour of seq is a solution in which a refuel bit is inverted.
            --------------
            Args :
                    seq: np.ndarray, valid encoding of solution
                    served: none
                    unserved: none
                    frozen_bits: list, contains frozen bits, not allowed to be touched during the search
            --------------
            Returns :
                    - valid solution encoding
                    - cost of solution returned

        """

        cost_heli, cost_pen, _ = self.compute_cost(seq)

        cost = cost_pen
        for h in cost_heli:
            cost += cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]


        k = 3
        i = 0
        queue = [(-cost, -1, seq)] #heapq
        for rf in self.ind_type_seq["Refuel"]:
            if rf in frozen_bits:
                #this is when we freeze some parts of the solution.
                continue

            h = rf // (2 * len(self.r))
            if h == (rf - 1) // (2 * len(self.r)) and seq[rf - 1] == 0:
                continue
            self.move += 1
            cand = seq.copy()
            cand[rf] = 1 - cand[rf]
            feas = self.check_refuel_ins(cand, rf)

            if not(feas):
                continue

            h_idx = rf // (2 * len(self.r))
            h = self.helicopters[h_idx]
            op_cand, viol_cand = self.trace_cost_heli(cand, h_idx, self.carac_heli[h]['start'], self.carac_heli[h]['init_fuel'],
                                                        self.carac_heli[h]['conso_per_minute'], self.carac_heli[h]['fuel_cap'], self.memo_cost_nb, self.cost_rf_nb,
                                                        self.chain_rf_nb, self.chain_nb, self.req_origin, self.req_dest, self.fly_time_nb,
                                                        self.h_idx_min, self.h_idx_max, self.len_encod, self.mintakeoff)
            cost_cand_heli = op_cand + self.pen_fuel * viol_cand
            cost_cand = cost - (cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]) + cost_cand_heli
            if cost_cand < np.inf:
                if len(queue) < k:
                    i += 1
                    heapq.heappush(queue, (-cost_cand, i, cand))
                else:
                    i += 1
                    _ = heapq.heappushpop(queue, (-cost_cand, i, cand))

        np.random.shuffle(queue)

        return queue[0][2], -queue[0][0]

    def rm_rf_neigh(
                    self,
                    seq: np.ndarray,
                    served: Any = None,
                    unserved: Any = None,
                    frozen_bits: List = []):
        """ Wrapper for the rm rf neighbourhoods.

            --------------
            Args :
                    seq: np.ndarray, valid encoding of solution
                    served: list, contains served requests in seq
                    unserved: list, contains unserved requests in seq
                    frozen_bits: list, contains frozen bits, not allowed to be touched during the search
            --------------
            Returns :
                    - valid solution encoding
                    - cost of solution returned

        """
        #getting cost in numba
        cost_heli, cost_pen, _ = self.compute_cost(seq)
        cost = cost_pen
        for h in cost_heli:
            cost += cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]

        cost_heli_nb = numba.typed.Dict.empty(
                        key_type=numba.typeof('h1'),
                        value_type= numba.typeof(1.),
                        )

        for h in cost_heli:
            cost_heli_nb[h] = float(cost_heli[h][0] + self.pen_fuel * cost_heli[h][1])

        #setting variables for fast swap neigh
        to_freeze = frozen_bits != []
        np.random.shuffle(served)
        id_res = np.random.randint(3)
        self.queue_rm_rf.clear()
        self.queue_rm_rf.append((float(cost), seq))

        q = self.rm_rf_neigh_fast(seq,
                                numba.typed.List(served),
                                numba.typed.List([-6]),
                                len(self.r),
                                self.len_encod,
                                self.served_by_fast,
                                self.get_reverse_indice,
                                self.remove_request_fast,
                                numba.typed.List(self.helicopters),
                                self.add_request_jit,
                                self.time_compatible,
                                self.refuel_compatible,
                                self.memo_cost_nb,
                                self.cost_rf_nb,
                                self.chain_rf_nb,
                                self.chain_nb,
                                self.req_origin,
                                self.req_dest,
                                self.fly_time_nb,
                                self.h_idx_min,
                                self.h_idx_max,
                                self.mintakeoff,
                                self.pen_fuel,
                                self.H,
                                self.trace_cost_heli_jit,
                                self.check_insertion_fast,
                                self.queue_rm_rf,
                                cost_heli_nb,
                                self.heli_start,
                                self.table_carac_heli,
                                cost, to_freeze, id_res, self.move, U.sort_ins,
                                self.check_refuel_ins
                              )

        return q

    #@staticmethod
    def rm_rf_neigh_fast(self,seq: np.ndarray,
                served,
                frozen_bits,
                len_r,
                len_encod,
                served_by_fast,
                get_reverse_indice,
                remove_request_fast,
                helicopters,
                add_request_jit,
                time_compatible,
                refuel_compatible,
                memo_cost_nb,
                cost_rf_nb,
                chain_rf_nb,
                chain_nb,
                req_origin,
                req_dest,
                fly_time_nb,
                h_idx_min,
                h_idx_max,
                mintakeoff,
                pen_fuel,
                H, trace_cost_heli, check_insertion_fast, queue_swap,
                cost_heli,
                heli_start,
                table_carac_heli, cost, to_freeze, id_res, move, sort_ins, check_refuel_ins
              ):
        """ Operator for the swap path neighbourhoods. Compute all neighbours of seq and returns a random one among
            the 3 best.
            A neighbour of seq is a solution in which 2 requests are swapped : if r1 is served by h1 in seq and r2 by h2, the
            neighbour is defined as the solution in which r1 is served by h2 and r2 is served by h1.
            --------------
            Args :
                    served: list of served requests, contains ints
                    frozen_bits: list of bits that cannot be touched, contains int
                    len_r: int, number of requests in problem instance
                    len_encod: int, length of encoding
                    served_by_fast:  method of self.
                    get_reverse_indice: method of self.
                    remove_request_fast: method of self.
                    helicopters: list of helictoper ids numba typed
                    add_request_jit: method of self.
                    time_compatible: numba dict, contains all requests that can be chained
                    refuel_compatible: numba dict, contains all requests that be chained with refuel inbetween
                    memo_cost_nb: numba dict, contains memoized costs
                    cost_rf_nb: numba dict, contains cost for any heli to start by serving a request with refuel
                    chain_rf_nb: numba dict, contains cost of chaining any pair of requests with refuel inbetween
                    chain_nb: numba dict, contains cost of chaining any pair of requests
                    req_origin: numba dict, contains request origin
                    req_dest: numba dict, contains request destination
                    fly_time_nb: numba dict, contains fly times
                    h_idx_min: int, min index of h in solution encoding
                    h_idx_max: int max index of h in solution encoding
                    mintakeoff: float, minimum amount of fuel required to takeoff
                    pen_fuel: float, fuel penalty term
                    H: int, number of heli
                    trace_cost_heli: method of self.
                    check_insertion_fast: method of self.
                    queue_swap: numba list to use for insertion sort of cands
                    cost_heli: numba dict, contains cost and violation for each heli
                    heli_start: np.ndarray, contains starting locations of each heli
                    table_carac_heli: np.ndarray, contains helicopters caracs
                    cost: float, cost of actual solution
                    to_freeze: bool, true iif frozen_bits is not empty
                    id_res: int between 0 and 2, for random selection of neighbour at the end of function
                    move : int, number of moves made so far
                    sort_ins: function insertion sort
                    check_refuel_ins: method of self.
            --------------
            Returns :
                    - valid solution encoding
                    - cost of solution returned

        """


        k_max = 3

        served_by_table = served_by_fast(seq, len_r, len_encod)


        for k in range(len(served)):
                r1 = served[k]

                #-- checking for frozen parts
                if to_freeze and get_reverse_indice(r1-1, len_encod, served_by_table[int(r1)-1]) in frozen_bits:
                    continue

                #removing request

                h_i = served_by_table[int(r1)-1]
                ind_r1_rm = get_reverse_indice(r1-1, len_encod, 0)
                ind_r1 = get_reverse_indice(r1-1, len_encod, int(h_i))
                #getting prev one in path of h if it was a refuel operation
                i = 1
                prev = None
                while (ind_r1 - i) // len_encod == h_i:
                    if seq[int(ind_r1 - i)] == 1:
                        prev = ind_r1 - i
                        break
                    i += 1
                if prev is None or prev % 2 == 0:
                    continue

                if to_freeze and prev in frozen_bits:
                    continue

                move = move + 2
                cand, h_i = remove_request_fast(seq, ind_r1_rm, len_encod, H)
                h = helicopters[h_i]

                #Adding refuel after prev

                cand[int(prev)+1] = 1

                feas =  check_refuel_ins(cand, int(prev)+1)

                if not(feas):
                    continue

                op_cand, viol_cand = trace_cost_heli(cand, h_i, heli_start[h_i], table_carac_heli[h_i, 4],
                                                        table_carac_heli[h_i, 2], table_carac_heli[h_i, 3],
                                                        memo_cost_nb, cost_rf_nb,
                                                        chain_rf_nb, chain_nb, req_origin, req_dest, fly_time_nb,
                                                        h_idx_min, h_idx_max, len_encod, mintakeoff)

                cand_cost_h = op_cand + pen_fuel * viol_cand

                cost_cand = cost - (cost_heli[h]) + cand_cost_h + self.pen_unserved

                if cost_cand < np.inf:
                    len_q = len(queue_swap)
                    queue_swap = sort_ins(len_q, k_max, queue_swap, cand, cost_cand)



        if id_res >= len(queue_swap):
            return queue_swap[-1][1], queue_swap[-1][0], move
        return queue_swap[id_res][1], queue_swap[id_res][0], move


    def add_request(
                    self,
                    seq: np.ndarray,
                    req: Tuple[str, str, int],
                    h: str) -> np.ndarray:
        """ Add request req to helicopter h in seq.
            --------------
            Args :
                  seq: np.ndarray, encoding of a valid solution
                  req: tuple, request
                  h: str, id of helicopter
            --------------
            Returns :
                    np.ndarray, encoding of valid solution

        """

        idrh = self.reverse_indices[h][req]
        seq_copy = seq.copy()
        seq_copy[idrh] = 1
        return seq_copy

    @staticmethod
    @njit(cache=True)
    def remove_request_fast(seq, ind_req, len_encod, H):
        """ Removes request req from solution seq. JITTED
            --------------
            Args :
                    seq: np.ndarray, encoding of a valid solution in which req is served
                    ind_req: int, id of req to be removed
                    len_encod: int, length of encoding
                    H: int, number of helicopters in problem instance
            --------------
            Returns :
                    updated solution encoding in which req is not served
                    helicopter id which served req prior to removal

        """
        seq_copy = seq.copy()

        for ht in range(H):
            if seq_copy[ind_req + ht * len_encod] == 1:
                h = ht
                idr = ind_req + ht * len_encod

        seq_copy[idr] = 0
        #remove the refuels that have become inconsistent with the structure due to this removal
        if (idr + 1) // len_encod == h:

          seq_copy[int(idr + 1)] = 0  #remove the next refuel if it was there
        #remove first refuel of the chain if there only one bit set to one.

        # ---- remove refuel that was just before if req was the last request served
        #getting prev one in path of h if it was a refuel operation
        i = 1
        prev = None
        while (idr - i) // len_encod == h:

            if seq_copy[int(idr - i)] == 1:
                prev = idr - i
                break
            i += 1
        if prev is None:
            return seq_copy, h
        #getting next one in path of h : if None, then req was the last to be served
        i = 1
        nxt = None
        while (idr + i) // len_encod == h:

            if seq_copy[int(idr + i)] == 1:
                nxt = idr + i
                break
            i += 1


        if prev is not None and prev%2 == 0 and nxt is None:
            seq_copy[int(prev)] = 0

        return seq_copy, h

    def remove_request(
                        self,
                        seq: np.ndarray,
                        req) -> Tuple[np.ndarray, str]:
        """ Removes request req from solution seq.
            --------------
            Args :
                    seq: np.ndarray, encoding of a valid solution in which req is served
                    req: tuple, request to be removed
            --------------
            Returns :
                    updated solution encoding in which req is not served
                    helicopter id which served req prior to removal

        """
        seq_copy = seq.copy()
        for ht in self.helicopters:
            if seq_copy[self.reverse_indices[ht][req]] == 1:
                h = ht
                idr = self.reverse_indices[ht][req]


        seq_copy[idr] = 0
        #remove the refuels that have become inconsistent with the structure due to this removal
        if idr+1 in self.assign[h]:
          seq_copy[idr + 1] = 0  #remove the next refuel if it was there
        #remove first refuel of the chain if there only one bit set to one.

        # ---- remove refuel that was just before if req was the last request served
        #getting prev one in path of h if it was a refuel operation
        i = 1
        prev = None
        while idr - i in self.assign[h]:
            if seq_copy[idr - i] == 1:
                prev = idr - i
                break
            i += 1

        if prev is None:
            return seq_copy, h
        #getting next one in path of h : if None, then req was the last to be served
        i = 1
        nxt = None
        while idr + i in self.assign[h]:
            if seq_copy[idr + i] == 1:
                nxt = idr + i
                break
            i += 1

        if prev is not None and "Ref" in self.indices[prev] and nxt is None:
            seq_copy[prev] = 0


        return seq_copy, h


    @staticmethod
    @njit
    def update_served_status_fast(seq, len_r, len_encod) -> Tuple[List, List]:
            """ Compute which requests are served/unserved in seq. JITTED
                --------------
                Args :
                        seq: np.ndarray, valid solution encoding
                        len_r: int, number of request
                        lend_encod: int, length of encoding
                --------------
                Returns :
                        - unserved requests
                        - served requests

            """
            unserved = [i for i in range(1, 1 + len_r)]
            served = []
            for i in range(1, len(seq) + 1, 2):
                #iterate over indices of requests only
                if seq[i] == 1:
                    req_idx = ((i % len_encod) + 1) / 2
                    served.append(int(req_idx))
                    unserved.remove(req_idx)
            return unserved, served

    @staticmethod
    @njit
    def served_by_fast(seq, len_r, len_encod):
        """ JITTED. Get tables indicating which heli serves which request in seq

            Args:
                seq: np.ndarray, solution encoding
                len_r: int, number of requests
                len_encod: int, length of encoding

            Returns:
                res: np.ndarray

        """
        res = - np.ones(len_r)
        for i in range(1, len(seq) + 1, 2):
            #iterate over indices of requests only
            if seq[i] == 1:
                req_idx = ((i % len_encod) + 1) / 2
                heli_idx = i // len_encod
                res[int(req_idx - 1)] = int(heli_idx)
        return res

    @staticmethod
    @njit
    def get_reverse_indice(req_idx, len_encod, h_idx):
        """ find indices in solution encoding that correspond to helictoper h_idx serving req req_idx.

            Args:
                req_idx:int, request id
                len_endod: int, length of encoding
                h_idx: int, helicopter id.

            Returns:
                int, indice in the solution encoding.

        """
        return int(h_idx * len_encod + (int(req_idx) + 1) * 2 - 1)

    @staticmethod
    @njit
    def add_request_jit(
                    seq: np.ndarray,
                    req_idx, len_encod,
                    h_idx, get_reverse_indice) -> np.ndarray:
            """ JITTED VERSION. Add request req to helicopter h in seq.
                --------------
                Args :
                      seq: np.ndarray, encoding of a valid solution
                      req_idx: int, request id
                      len_encod: int, length of encoding
                      h_idx: int, heli id
                      get_reverse_indices: method of self.
                --------------
                Returns :
                        np.ndarray, encoding of valid solution

            """
            idrh = get_reverse_indice(req_idx, len_encod, h_idx)
            seq_copy = seq.copy()
            seq_copy[idrh] = 1
            return seq_copy



    def update_served_status(
                            self,
                            seq: np.ndarray) -> Tuple[List, List]:
        """ Compute which requests are served/unserved in seq
            --------------
            Args :
                    seq: np.ndarray, valid solution encoding
            --------------
            Returns :
                    - unserved requests
                    - served requests

        """

        served = [self.indices[i] for i in self.ind_type_seq['Request'] if seq[i]==1]
        unserved = set(self.r.keys()) - set(served)
        return list(unserved), list(served)

    def init_heuristic(self) -> np.ndarray:
        """ Constructs an initial solution based on heuristic : request are sequentially inserted in sol until not possible.
            --------------
            Args :
                    None
            --------------
            Returns :
                    np.ndarray, encoding of valid solution

        """
        served = []
        init_sol = self.empty_sol.copy()

        for h in self.assign:

            routeh = []
            start = self.carac_heli[h]["start"]
            for i in self.assign[h]:
                if routeh == [] and (start, self.indices[i]) in self.time_compatible and not(self.indices[i] in served):
                    init_sol[i] = 1
                    served.append(self.indices[i])
                    routeh.append(self.indices[i])

                elif routeh and (routeh[-1], self.indices[i]) in self.time_compatible and not(self.indices[i] in served):
                    init_sol[i] = 1
                    served.append(self.indices[i])
                    routeh.append(self.indices[i])

        return init_sol

    def init_heuristic_random(self) -> np.ndarray:
        """ Constructs a random initial solution.
            --------------
            Args :
                    None
            --------------
            Returns :
                    np.ndarray, encoding of valid solution

        """
        served = []
        init_sol = self.empty_sol.copy()
        served_heli = {h:[] for h in self.helicopters}
        served = []
        n = 2*len(self.r)

        o = self.ind_type_heli["h1"]["Request"].copy()

        np.random.shuffle(o)
        for i in o:

            if i in served:
                continue
            #get feasible helis
            cand_h = []
            for h in self.helicopters:
                if served_heli[h] == []:
                    h_i = self.helicopters.index(h)
                    h_id = -int(h.replace('h', ''))
                    if (h_id, i + h_i * n) in self.time_compatible:
                        cand_h.append(h)
                else:
                    h_i = self.helicopters.index(h)
                    if (self.reverse_indices[h][served_heli[h][-1]], i + h_i * n) in self.time_compatible:
                        cand_h.append(h)
            #choose random assignment - uniformly
            if not(cand_h):
                #no more insertion possible
                break
            h_elected = np.random.choice(cand_h)
            heli = self.helicopters.index(h_elected)
            ind = i + heli * n
            init_sol[ind] = 1
            served_heli[h_elected].append(self.indices[i])
            served.append(i)
        return init_sol

    def compute_imbalance(
                          self,
                          sol: np.ndarray) -> float:
        """ Compute the imbalance of helicopters in sol, i.e. how different the loads of each helicopter are (in numbers).

            --------------
            Args :
                    sol: np.ndarray, encoding of valid solution
            --------------
            Returns :
                    float, imbalance metric

        """
        nb_s = []
        for h in self.helicopters:
            nsh = 0
            for bi in self.assign[h]:
                if not ("Ref" in self.indices[bi]):
                    nsh += sol[bi]
            nb_s.append(nsh)
        std = np.std(nb_s)
        mu = np.mean(nb_s)
        if mu == 0 or std == 0:
            return 0
        return std / mu

    def viz_graph(
                  self,
                  arcs: Dict[str, Tuple[Tuple[str, int], Tuple[str, int]]],
                  A: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                  A_s: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                  A_g: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                  A_f: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                  colors: Dict[str, str],
                  note: str = "") -> Any:
        """ Viz functions for the time expanded network, schedule is represented on graph.
          --------------
          Args :
                arcs : dict variable, contains arcs present in each helicopter's path
                A : list, contains all arcs
                A_g : list, contains all refuelling arcs
                A_s : list, contains all service arcs
                colors : dict, colors to represent helicopters on graph
                notes : str

          --------------
          Returns :
                fig : matplotlib figure to be plotted/saved

        """
        img=mpimg.imread('/project/src/image/icon.png')
        G = nx.DiGraph()
        for n in self.nodes:
            G.add_node(n, pos=(n[1], self.locations.index(n[0])))
        G.add_edges_from(A)
        for h in self.helicopters:
            for e in A_g:
                if e in arcs[h]:
                    G.add_edge(e[0], e[1], image=img, size=0.09)

        pos = nx.get_node_attributes(G, 'pos')
        fig, ax = plt.subplots(figsize=(19, 7))
        plt.title(f"Graph Schedule, Rand. {len(self.r)}.H{len(self.helicopters)} - {note}")

        nx.draw_networkx_nodes(G, pos, node_size=5, ax=ax, node_color='black')
        for h in self.helicopters:
            if h in arcs:
                nx.draw_networkx_edges(G,pos,
                                      edgelist=[e for e in A if e in arcs[h] and not(e in A_f)],
                                      width=4, alpha=1, edge_color=colors[h])
                nx.draw_networkx_edges(G,pos,
                                      edgelist=[e for e in A if e in arcs[h] and e in A_f],
                                      width=1, alpha=1, edge_color=colors[h], style="dotted")

        nx.draw_networkx_edges(G,pos,
                            edgelist=[e for e in A_s],
                            width=1, alpha=0.5, edge_color='g')

        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlabel("Time")
        plt.yticks([i for i in range(len(self.locations))], self.locations)
        plt.xticks([i for i in range(len(self.T))][::30], ['{:02d}:{:02d}'.format(*divmod(420 + i, 60)) for i in range(len(self.T))][::30])

        # add images on edges
        ax2=plt.gca()
        fig2=plt.gcf()
        label_pos = 0.5 # middle of edge, halfway between nodes
        trans = ax2.transData.transform
        trans2 = fig2.transFigure.inverted().transform
        imsize = 0.1  # this is the image size
        rf = []
        for h in self.helicopters:
            rf += [e for e in A_g if e in arcs[h]]
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


    def descibe_sol(
                    self,
                    sol: np.ndarray,
                    A_s: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                    A_g: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                    A: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                    notes: str = "",
                    verbose: bool = True) -> Dict:
        """ Prints routing of solution sol in plain english detailing the path of eahc helicopter.
            --------------
            Args :
                    sol: np.ndarray, encoding of valid solution
                    A_s: list, contains all service arcs
                    A_g: list, contains all refuelling arcs
                    A: list, contains all arcs
                    notes: str, to print before verbose
                    verbose: bool, whether to print details about sol
            --------------
            Returns :
                    dict, contains log for each helicopter

        """
        if verbose:
            print(notes)
        unserved, served = self.update_served_status(sol)
        cost = 0
        paths_sol = {}
        for h in self.helicopters:
            if verbose:
                print("")
            costh, log = self.compute_cost_heli(sol, h)
            path = self.read_log(log, h, A_s, A_g, A, verbose=verbose)
            paths_sol[h] = path
            cost += costh[0] + self.pen_fuel * costh[1]
            if verbose:
                print("---------")
        if verbose:
            print("")
            print(f"Operationnal cost : {cost} - Penalty for unserved demands : {self.pen_unserved * len(unserved)} - Total : {cost + self.pen_unserved * len(unserved)}")
            print(f"{len(served)} requests are served out of {len(self.r)}")
        return paths_sol

    def get_arcs(
                self,
                paths_sol: Dict[str, List[Tuple[str, int]]]) -> Dict[str, List[Tuple[Tuple[str, int], Tuple[str, int]]]]:
        """ Gets arcs to plot on the TEN to represent solution
            --------------
            Args :
                    paths_sol: dict, contains path for several helicopters, output of describe_sol()
            --------------
            Returns :
                    arcs : list of arcs for the TEN

        """
        arcs = {}
        for h in paths_sol:
            ar = []
            for i in range(1, len(paths_sol[h])):
                ar.append((paths_sol[h][i-1], paths_sol[h][i]))
            arcs[h] = ar
        return arcs

    def viz_convergence(
                        self,
                        cache_cost_best: List[float],
                        cache_cost_current: List[float],
                        note: str = ""):
        """ Plot cost over time, both best and current issued from VNS and saves fig to /image.
            --------------
            Args :
                    cache_cost_best: list
                    cache_cost_current: list
            --------------
            Returns :
                    None

        """
        plt.figure(figsize=(14,7))
        plt.plot(cache_cost_best, label="Best costs")
        plt.plot(cache_cost_current, label="Explored costs")
        plt.xlabel("Iterations")
        plt.title(f"Cost evolution - R{len(self.r)}. H{len(self.helicopters)} - {note}")
        plt.legend()
        plt.savefig("image/vns_convergence.png")






class MILP2():
  def __init__(self, helicopters, nodes, r, A, locations, T, carac_heli, sinks, parking_fee,
              A_w, A_s, A_f, A_g, beta, fees, fly_time, min_takeoff, refuel_prices, capacity, refuel_times, pen):
    """ Implement a MILP to solve the model under new assumption of no intermediary destinations : based on encoding of MH.
      --------------
      Params :
              helicopters : list, id of helicopters available to use
              nodes : list, contains all timed nodes
              r : list, contains all requests
              A : list, contains all arcs, waiting, service and deadhead
              locations : list, id of different heliports
              T : list, contains all timesteps (assumed to be separated by one minute)
              carac_heli : dict, contains caracteristics of helicopters
              sinks : dict, maps each helicopter to a unique sink node.
              parking_fee : dict
              A_s : list, service arcs
              A_w : list, waiting arcs
              A_f : list, deadhead arcs
              beta : float, gain from serving a request in dollars - This will become a dict
              to give different values to different requests.
              fees : dict, gives landing fee for each location
              fly_time : dict, give fly time in minutes (rounded) between locations.


      --------------
      Returns :
              None

    """
    self.helicopters = helicopters
    self.A = A
    self.nodes = nodes
    self.r = r
    self.end = U.connect_sink([sinks[h] for h in helicopters], locations, T)
    self.locations = locations
    self.T = T
    self.carac_heli = carac_heli
    self.sinks = sinks
    self.Ah = list(itertools.product(A + self.end, helicopters))
    self.costs = U.create_cost(A_w, A_s, A_f, A_g, self.end, carac_heli, beta, fees, helicopters, fly_time, refuel_prices)
    self.parking_fee = parking_fee
    self.A_g = A_g
    self.fly_time = fly_time
    self.min_takeoff = min_takeoff
    self.A_w = A_w
    self.A_s = A_s
    self.A_f = A_f
    self.capacity = capacity
    self.couples = U.chain_requests(r, helicopters, carac_heli)
    self.cost_chain = U.cost_chain(self.couples, helicopters, carac_heli, fly_time, fees)
    self.req = set([p[0] for p in self.couples])
    self.refuel_prices = refuel_prices
    self.refuel_times = refuel_times
    self.parking_fee = parking_fee
    self.pen = pen


  def build_model(self):
    """ Build the Integers Linear Program instance.
      --------------
      Params :
              None
      --------------
      Returns :
              None

    """
    # First creates the master problem variables of whether
    self.y = plp.LpVariable.dicts("hrr", (self.helicopters, self.req, self.req), 0, 1, plp.LpInteger)

    #Create auxiliary variable to monitor other constraint, i.e. parking time, fuel
    self.fa = plp.LpVariable.dicts("rfa", (self.helicopters, self.req, self.req), 0, 1, plp.LpInteger)

    self.ta = plp.LpVariable.dicts("ta", (self.helicopters, self.req, self.req), 0, None, plp.LpInteger)
    self.tb = plp.LpVariable.dicts("tb", (self.helicopters, self.req, self.req), 0, None, plp.LpInteger)
    self.fy = plp.LpVariable.dicts("fy", (self.helicopters, self.req, self.req), 0, 1, plp.LpInteger)

    self.pa = plp.LpVariable.dicts("pa", (self.helicopters, self.req, self.req), 0, 1, plp.LpInteger)
    self.pb = plp.LpVariable.dicts("pb", (self.helicopters, self.req, self.req), 0, 1, plp.LpInteger)
    self.pab = plp.LpVariable.dicts("pab", (self.helicopters, self.req, self.req), 0, 1, plp.LpInteger)


    self.nb_un = plp.LpVariable("unserved", lowBound = 0, cat = plp.LpInteger)

    self.v = plp.LpVariable.dicts("FuelLvl", (self.helicopters, self.locations, self.T), 0, 1100, plp.LpContinuous) #this is the fuel level variable
    self.z = plp.LpVariable.dicts("zede", (self.helicopters, self.req), 0, None, plp.LpInteger)
    #Instantiate the model
    self.model = plp.LpProblem(name="optim", sense=plp.LpMinimize)

    #Objective function


    self.model += plp.lpSum([self.y[h][p1][p2] * self.cost_chain[h][p1][p2] + self.fy[h][p1][p2] * self.refuel_prices[p1[1]] + self.parking_fee[p1[1]] * self.pab[h][p1][p2] * (p1[1] == p2[0]) + (p1[1] != p2[0]) * (self.parking_fee[p1[1]] * self.pa[h][p1][p2] + self.parking_fee[p2[0]] * self.pb[h][p1][p2]) for h in self.helicopters for p1 in self.req for p2 in self.req if p1 != p2 and p2[2] + p2[3] > 0]) + self.pen * self.nb_un, "Total Costs"


    #Adding constraints
    self.model += self.nb_un == plp.lpSum([1 - plp.lpSum([self.y[h][p1][p2] for h in self.helicopters for p1 in self.req if p1 != p2]) for p2 in self.req if p2[0] != p2[1]])


    for p1 in self.req:
       for p2 in self.req:
           self.model += plp.lpSum([self.y[h][p1][p2] for h in self.helicopters]) <= 1



    for p2 in self.req:

        if p2[0] != p2[1]:

            self.model += plp.lpSum([self.y[h][p1][p2] for h in self.helicopters for p1 in self.req if p1 != p2]) <= 1 #maxsat mode
        for h in self.helicopters:
          self.model += plp.lpSum([self.y[h][p2][p1] for p1 in self.req if p1 != p2]) <= 1

    for h in self.helicopters:

        starth = [r for r in self.req if r[0] == r[1] == self.carac_heli[h]["start"]][0]
        self.model += self.v[h][starth[0]][0] == self.carac_heli[h]["init_fuel"]

        for p1 in self.req:
            for p2 in self.req:

                if p1 != p2:

                    self.model += self.tb[h][p1][p2] >= 0
                    self.model += self.ta[h][p1][p2] >= 0
                    self.model += self.tb[h][p1][p2] <= p2[2] - p1[3] - self.fa[h][p1][p2] * self.refuel_times[p1[1]] - self.fly_time[(p1[1], p2[0])] - self.ta[h][p1][p2]  + (1 - self.y[h][p1][p2]) * (max(self.T) + max(list(self.fly_time.values())))
                    self.model += self.tb[h][p1][p2] >= p2[2] - p1[3] - self.fa[h][p1][p2] * self.refuel_times[p1[1]] - self.fly_time[(p1[1], p2[0])] - self.ta[h][p1][p2]  - (1 - self.y[h][p1][p2]) * (max(self.T) + max(list(self.fly_time.values())))


                    conso = self.carac_heli[h]["conso_per_minute"] * self.fly_time[(p1[1], p2[0])]

                    self.model += self.ta[h][p1][p2] <= 1 - self.y[h][p1][p2] + (self.y[h][p1][p2]) * max(self.T)
                    self.model += self.tb[h][p1][p2] <= 1 - self.y[h][p1][p2] + (self.y[h][p1][p2]) * max(self.T)


                    self.model += self.fy[h][p1][p2] <= self.fa[h][p1][p2]
                    self.model += self.fy[h][p1][p2] <= self.y[h][p1][p2]
                    self.model += self.fy[h][p1][p2] >= self.fa[h][p1][p2] + self.y[h][p1][p2] - 1


                    self.model += self.y[h][p1][p2] <= plp.lpSum([self.y[h][starth][p] for p in self.req])

                    if p1 != starth:
                        self.model += self.y[h][p1][p2] <= plp.lpSum([self.y[h][p][p1] for p in self.req if p != p1])

                    #parking fees
                    #pa = indicate if idle time after p1 is greater than 15
                    #pb = indicate if idle time before p2 is greater than 15

                    #parking fee at p1
                    self.model += self.ta[h][p1][p2] - 15 <= (max(self.T) - 15) * self.pa[h][p1][p2]
                    self.model += 15 - self.ta[h][p1][p2] <= 15 * (1 - self.pa[h][p1][p2])

                    #parking fee at p2
                    self.model += self.tb[h][p1][p2] - 15 <= (max(self.T) - 15) * self.pb[h][p1][p2]
                    self.model += 15 - self.tb[h][p1][p2] <= 15 * (1 - self.pb[h][p1][p2])

                    #if p1 ends where p2 starts, parking fee ta + tb >= 15

                    self.model += self.ta[h][p1][p2] + self.tb[h][p1][p2] - 15 <= (max(self.T) - 15) * self.pab[h][p1][p2]
                    self.model += 15 - self.tb[h][p1][p2] - self.ta[h][p1][p2] <= 15 * (1 - self.pab[h][p1][p2])

                    self.model += self.pab[h][p1][p2] <= self.y[h][p1][p2]
                    self.model += self.pa[h][p1][p2] <= self.y[h][p1][p2]
                    self.model += self.pb[h][p1][p2] <= self.y[h][p1][p2]


                    #monitoring the fuel consumption
                    self.model += self.v[h][p2[0]][p2[2]]  >= self.fa[h][p1][p2] * (self.carac_heli[h]["fuel_cap"]) - conso - (1 - self.y[h][p1][p2]) * (2 * self.carac_heli[h]["fuel_cap"]) + self.z[h][p1]
                    self.model += self.v[h][p2[0]][p2[2]]  <= self.fa[h][p1][p2] * (self.carac_heli[h]["fuel_cap"]) - conso + (1 - self.y[h][p1][p2]) * (2 * self.carac_heli[h]["fuel_cap"]) + self.z[h][p1]

                    self.model += self.v[h][p1[1]][p1[3]] >= self.v[h][p1[0]][p1[2]] - self.carac_heli[h]["conso_per_minute"] * self.fly_time[(p1[0], p1[1])] - (1 - self.y[h][p1][p2]) * (self.carac_heli[h]["fuel_cap"] + self.carac_heli[h]["conso_per_minute"] * self.fly_time[(p1[0], p1[1])])
                    self.model += self.v[h][p1[1]][p1[3]] <= self.v[h][p1[0]][p1[2]] - self.carac_heli[h]["conso_per_minute"] * self.fly_time[(p1[0], p1[1])] + (1 - self.y[h][p1][p2]) * (self.carac_heli[h]["fuel_cap"] + self.carac_heli[h]["conso_per_minute"] * self.fly_time[(p1[0], p1[1])])



                    #enforce fuel constraint min at takeoff at p1+ to p2- and at p2- to p2+

                    if p1[1] != p2[0]:
                        self.model += self.v[h][p1[1]][p1[3]] >= self.y[h][p1][p2] * self.min_takeoff - self.fa[h][p1][p2] * self.carac_heli[h]["fuel_cap"]

                    self.model += self.v[h][p2[0]][p2[2]] >= self.y[h][p1][p2] * self.min_takeoff  - (1 - self.y[h][p1][p2]) * self.carac_heli[h]["fuel_cap"]


                    self.model += self.z[h][p1] >= self.v[h][p1[1]][p1[3]] - self.fa[h][p1][p2] *  self.carac_heli[h]["fuel_cap"] - (1 - self.y[h][p1][p2]) * self.carac_heli[h]["fuel_cap"]
                    self.model += self.z[h][p1] <= (1 - self.fa[h][p1][p2]) *  self.carac_heli[h]["fuel_cap"]
                    self.model += self.z[h][p1] <= self.v[h][p1[1]][p1[3]]




  def solve(self, max_time, opt_gap, verbose=1, verbose2 = True, warmstart=False):
    """ Solve the Integers Linear Program instance built in self.build_model().
      --------------
      Params :
              max_time : int, maximum running time required in seconds.
              opt_gap : float, in (0, 1), if max_time is None, then the objective value
              of the solution is guaranteed to be at most opt_gap % larger than the true
              optimum.
              verbose : 1 to print log of resolution. 0 for nothing.
      --------------
      Returns :
              Status of the model : Infeasible or Optimal.
              Infeasible indicates that all constraints could not be met.
              Optimal indicates that the model has been solved optimally.

    """
    start = time.time()
    self.model.solve(plp.PULP_CBC_CMD(maxSeconds = max_time, fracGap = opt_gap, msg = verbose, mip_start=warmstart, options=["randomCbcSeed 31"]))
    #Get Status
    if verbose2:
        print("Status:", plp.LpStatus[self.model.status])
        print("Total Costs = ", plp.value(self.model.objective))
        print("Solving CPU time : ", round(time.time() - start, 3), " seconds.")
    return plp.LpStatus[self.model.status], plp.value(self.model.objective), round(time.time() - start, 3)



  def convert_sol(self):
    """ Convert the solution computed by MILP to time space path format.
        Args:


        Returns:
            paths: dict. keys are helicopters and values list which contains the sequence of timed node that make the path of h.
            nb_uns: int, number of unserved requests

    """
    paths = {h:[] for h in self.helicopters}
    for h in self.helicopters:

      for p1 in self.req:
        for p2 in self.req:
          if p1 != p2 and self.y[h][p1][p2].varValue == 1:
              if self.fa[h][p1][p2].varValue == 1:
                  if p1[1] == p2[0]:
                      l = [(p1[1], p1[3]), (p1[1], p1[3] + self.refuel_times[p1[1]])]
                      l += [(p1[1], t) for t in range(l[-1][1] + 1, p2[2] + 5)]
                  else:
                      #print(p1, p2, ins.ta[h][p1].varValue, int(ins.tb[h][p2].varValue))
                      l = [(p1[1], p1[3])]
                      l += [(p1[1], p1[3] + self.refuel_times[p1[1]] + t) for t in range(int(self.ta[h][p1][p2].varValue) + 1)]
                      l += [(p2[0], p2[2] - t) for t in range(int(self.tb[h][p1][p2].varValue) + 1)]
                      l += [(p2[0], p2[2] + t) for t in range(5)]

              else:

                  if p1[1] == p2[0]:
                      l = [(p1[1], t) for t in range(p1[3], p2[2] + 5)]

                  else:

                      l = [(p1[1], p1[3] + t) for t in range(int(self.ta[h][p1][p2].varValue) + 1)]
                      l += [(p2[0], p2[2] - t) for t in range(int(self.tb[h][p1][p2].varValue) + 1)]
                      l += [(p2[0], p2[2] + t) for t in range(1, 5)]

              paths[h] += l
      if paths[h]:
        paths[h] = sorted(paths[h], key=lambda x: x[1])
        out = [a for a in self.A_s if a[0] == paths[h][-1]][0]
        paths[h] += [out[0], out[1]]

      #Get number of unserved demands :
      nb_uns = sum([1 - sum(filter(None, [self.y[h][p1][p2].varValue for h in self.helicopters for p1 in self.req if p1 != p2])) for p2 in self.req if p2[0] != p2[1]])

    return paths, nb_uns


  def read(self, path, h):
    """ Reads the paths of helicopter h to stdout under human readable way.

        Args:
                path: dict, as output by self.convert_sol
                h: str, helicopter
        Returns:
                None

    """
    fuel_level = self.carac_heli[h]["init_fuel"]
    print(h, " starts with ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), " % of fuel.")
    served = 0
    for i in range(1, len(path)):
        prev, succ = path[i-1], path[i]
        a = (prev, succ)
        if not(a in self.A):
            continue
        if prev[0] != succ[0]:
            #change in location
            fuel_level -= self.fly_time[(prev[0], succ[0])] * self.carac_heli[h]["conso_per_minute"]
            if a in self.A_s:
                served += 1
                print(h, " starts service in ", prev[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60)), "and finishes in ", succ[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60)), " - Fuel level at arrival: ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), "%", " - Minimum level reached " * (fuel_level < 325) )
            else:
                print(h, " leaves from ", prev[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60)), "and arrives in ", succ[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60)), " - Fuel level at arrival: ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), "%", " - Minimum level reached " * (fuel_level < 325))

        else:
            if a in self.A_g:
                fuel_level = self.carac_heli[h]["fuel_cap"]
                print(h, " starts refueling in ", prev[0], " at ",'{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60)), ", finishes at ", '{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60)), " - Fuel level at arrival: ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), "%", " - Minimum level reached " * (fuel_level < 325))
    print("")

  def get_arcs(self, paths_sol):
        """ Creates arcs for time space network

            Args:
                paths_sol: dict, as output by self.convert_sol

            Returns:
                arcs: dict, contains for each heli the list of arcs in its path, ordered.
        """
        arcs = {}
        for h in paths_sol:
            ar = []
            for i in range(1, len(paths_sol[h])):
                ar.append((paths_sol[h][i-1], paths_sol[h][i]))
            arcs[h] = ar
        return arcs

  def get_imbalance(self):
      """ Computes imbalance metric for the solution.
          Imbalance is defined as the ratio between the std and the mean of the following series:
            - number of served request by each helicopter.

      """
      nb_s = []
      for h in self.helicopters:
          nbsh = 0
          for p1 in self.req:
              for p2 in self.req:
                  nbsh += self.y[h][p1][p2].varValue
          nb_s.append(nbsh)
      mu = np.mean(nb_s)
      std = np.std(nb_s)
      if mu == 0 or std == 0:
        return 0

      return std / mu




