import numpy as np
import itertools
import operator
import matplotlib.pyplot as plt
import networkx as nx
import time
import utils as U
import matplotlib.image as mpimg
import heapq

#######################
# THIS MODULE IS NOT USED IN PRACTICE IN THE REPO.
# It is a try at using lagrange method for optimization in
# the SLS approach
########################


class MH2():
    def __init__(self, r, helicopters, carac_heli, refuel_time, refuel_price, locations, nodes, parking_fee,
                 landing_fees, fly_time, T, beta, mintakeoff, pen_fuel, A_s, A, A_g):
        """ THIS CLASS IS NOT USED IN PRACTICE IN THE REPO.
            It is kept because it is a try to another kind of optimization using
            Lagrange method. Documented in the section Ongoing works in the overleaf but the following code
            is not documented properly
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

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

    def init_encoding(self):
        """ Desc
            --------------
            Params :
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
        self.empty_sol = np.zeros(len(events))
        self.indices = indices
        self.assign = assign
        self.ind_type = ind_type
        self.ind_type_heli = ind_type_heli
        self.ind_type_seq = ind_type_seq
        self.connect_memo = {}
        self.rf_cons = [None] * 5
        self.rf_ev = [None] * 3


    def init_compatibility(self):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        self.time_compatible, self.refuel_compatible = U.preprocessing(self.r, self.fly_time, self.helicopters, self.carac_heli, self.refuel_time, self.locations)
        #print(self.time_compatible)


    def init_request_cost(self):
        """ Desc
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


    def feasible(self, arr):
        """ Test if instance is feasible or not : check for logical infeasibilies but not for
        fuel violations. Fuel violations will be penalized in local search.
        This function will test for :
            - Two different helicopters serving the same demand
            - Refuelling happening for nothing. Refuel slot can be taken in two cases :
            previous demand slot in 1 or the refuel slot correspond to the beginning of service, in which case
            at least 1 demand should be served.
            - Consecutive served demand are time-compatible
            - Consectuvie served demand are time compatible and refuelling is possible inbetween is refuel slot is taken.

        """
        #hs = arr.tostring()
        if not(hasattr(self, "time_compatible")):
            raise AttributeError("Compatibility is not initialized. Call init_compatibility() before using other methods.")
        if not(hasattr(self, "pen_unserved")):
            raise AttributeError("Unserved penalty is not initialized. Call init_request_cost() before using other methods.")
        #check for serving conflict
        tab = arr.reshape((len(self.helicopters), 2 * len(self.r)))

        if not (np.all(np.sum(tab[:, self.mask], axis=0) <= 1)):
            return False

        #check for unauthorized refuel or useless ones

        if not (np.all(tab[:, ~self.mask][:, 0] <= np.sum(tab[:, self.mask], axis=1))):
            #print("Useless refuel begin")
            return False
        if not (np.all(tab[:, ~self.mask][:, 1:] <= tab[:, self.mask][:, :-1])):
            #print("Useless refuel")
            return False

        for h in self.helicopters:
            #path = arr[self.assign[h]]
            path = tab[self.helicopters.index(h), :]
            #n = len(self.assign[h])
            start = self.carac_heli[h]["start"]
            if not(np.any(path)):
                continue

            #check for time-compatibility violations and refuel-time violations
            idx = np.where(path == 1)[0]
            if idx[-1] in self.ind_type["Refuel"]:
                #print("Refuel compat")

                return False#, None
            #if len(idx) == 1:
            if not( idx[0] in self.ind_type["Refuel"] ) and not((start, self.indices[idx[0]])) in self.time_compatible:
                #print("Time compat 1")

                return False  #, self.time_violation[(None, self.indices[idx[0]])]

            elif idx[0] in self.ind_type["Refuel"] and not((start, self.indices[idx[1]]) in self.refuel_compatible):
                #print("Refuel compat 1")

                return False#, self.refuel_violation[(None, self.indices[idx[1]])]

            for i in range(1, len(idx)):
                if idx[i-1] in self.ind_type["Refuel"]:
                    continue
                prev, succ = idx[i-1], idx[i]
                #print(prev, succ)
                if prev in self.ind_type["Request"] and succ in self.ind_type["Request"] and not((self.indices[prev], self.indices[succ]) in self.time_compatible):
                    #print("Time compat 2")

                    return False#, self.time_violation[(self.indices[prev], self.indices[succ])]
                if prev in self.ind_type["Request"] and succ in self.ind_type["Refuel"] and not((self.indices[prev], self.indices[idx[i+1]]) in self.refuel_compatible):
                    #print("Refuel compat 2")

                    return False#, self.refuel_violation[(self.indices[prev], self.indices[idx[i+1]])]
        return True#, None


    def evaluate_link(self, con_points, h, entry_fuel, getfuel=False):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """

        warnings.warn("Function is depreciated. Should not be used")
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


    def connects(self, n, m, h, entry_fuel):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """


        i, t1 = n
        j, t2 = m
        #con_points = []
        if i == j:
            straight_path = [(i, t1), (i, t2)]
            #c_straight = self.evaluate_link(straight_path, h, entry_fuel)
            c_straight = (t2 - t1 > 15) * self.parking_fee[i]
            return straight_path, [c_straight, 0], entry_fuel

        else:
            free_delta = t2 - t1 - self.fly_time[(i, j)]
            if free_delta < 0:
                #con_points = []
                return [], [np.inf, 0], entry_fuel
            else:
                if free_delta <= 15 * 2:
                    ta = t1 + int(free_delta/2)
                else:
                    ta = t1 + free_delta * (self.parking_fee[i] < self.parking_fee[j]) + 1 * (self.parking_fee[j] <= self.parking_fee[i])

                link = [(i, t1), (i, ta), (j, ta + self.fly_time[(i, j)]), (j, t2)]
                #c, fuel = self.evaluate_link(link, h, entry_fuel, getfuel=True)
                p_fee = (free_delta > 15 * 2) * min(self.parking_fee[i], self.parking_fee[j])
                fuel_viol = max(0, self.mintakeoff - entry_fuel) #self.pen_fuel * (self.mintakeoff - entry_fuel) * (entry_fuel <= self.mintakeoff)
                c = self.fly_time[(i, j)] * self.carac_heli[h]["cost_per_min"] + self.landing_fee[j]
                c += p_fee
                newfuel = entry_fuel - self.fly_time[(i, j)] * self.carac_heli[h]["conso_per_minute"]

                return link, [c, fuel_viol], newfuel


    def refuel_slot(self, n, m, h, entry_fuel):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        i, t1 = n
        j, t2 = m
        link_path, cost, newfuel = self.connects((i, t1 + self.refuel_time[i]), (j, t2), h, self.carac_heli[h]["fuel_cap"])
        cost[0] += self.refuel_price[i]
        #print("Rf link =", cost)
        #return route_fuel, cost, newfuel
        rf_cons = [(i, t1), ('Ref', i, t1)] + link_path
        return rf_cons, cost, newfuel

    def compute_cost_heli(self, arr, h, logfuel=False, log=True):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        self.nb_evaluation += 1
        arr_heli = arr[self.assign[h]]
        #ones in the chain of helicopter h
        ones_index = list(np.where(arr_heli == 1)[0])

        #self.ind_type["Request"]
        ones_index = [i for i in self.ind_type["Request"] if i in ones_index ]
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
        #print("Connecting ", n, " to ", m, " starting fuel = ", fuel, " || Current of connection ", c, " || Cum cost = ", cost)
        fuel_viol += max(0, self.mintakeoff - newfuel) #(newfuel <= self.mintakeoff) * self.pen_fuel * (self.mintakeoff - newfuel)
        newfuel -= self.carac_heli[h]["conso_per_minute"] * self.fly_time[(self.indices[succ][0], self.indices[succ][1])]
        cost += self.service_cost[h][self.indices[succ]]
        #print("Serving from ", self.indices[succ][0], " to ", self.indices[succ][1] , "||New Cum cost = ", cost, " || Fuel after service ", newfuel)
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
            #print("Connecting ", n, " to ", m, " starting fuel = ", newfuel, " || Current of connection ", c, " || Cum cost = ", cost)
            fuel_viol += c[1]
            fuel_viol += max(0, self.mintakeoff - newfuel)
            #cost += (newfuel <= self.mintakeoff) * self.pen_fuel * (self.mintakeoff - newfuel)
            newfuel -= self.carac_heli[h]["conso_per_minute"] * self.fly_time[(self.indices[nxt][0], self.indices[nxt][1])]
            cost += self.service_cost[h][self.indices[nxt]]
            #print("Serving from ",self.indices[nxt][0], " to ", self.indices[nxt][1] , "||New Cum cost = ", cost, " || Fuel after service ", newfuel)
            if log:
                logs += points
          if log:
            logs += [req_succ[-1][1]]
            return [cost, fuel_viol], logs
          else:
            return [cost, fuel_viol]

    def compute_cost(self, seq, getlog=True):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        feas = self.feasible(seq)
        if not(feas):
            if getlog:
                return {h: [np.inf, 0] for h in self.helicopters}, 0, {}
            else:
                return np.inf
        cost = 0
        if getlog:
            cost_heli = {}
            cache_heli = {}
        for h in self.helicopters:
            if getlog:
                c, log = self.compute_cost_heli(seq, h)
                cost_heli[h] = c
                cache_heli[h] = log
            else:
                op_cost, fuel_viol = self.compute_cost_heli(seq, h, log=False)
                cost += op_cost + self.pen_fuel * fuel_viol


        if getlog:
            unserved, _ = self.update_served_status(seq)
            cost_pen = len(unserved) * self.pen_unserved
            return cost_heli, cost_pen, cache_heli
        else:
            return cost


    def read_log(self, log, h, A_s, A_g, A):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """

        #build path from log
        if len(log) == 0:
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
                #print((current[1], current[2]), nxt, " Refueling")
                path += [(current[1], current[2]), nxt]
                continue
            dum = (current[0], current[1] + 4)
            if (dum, nxt) in A_s:
                #print(current, nxt, " Service")
                path += [(current[0], current[1] + t) for t in range(1, 5)]
                path += [nxt]
                continue

            if current[0] == nxt[0]:
                #print(current, nxt, " Straight")
                path += [(current[0], current[1] + t) for t in range(int(nxt[1] - current[1]))] + [nxt]
                continue

            if nxt[1] - current[1] == self.fly_time[(current[0], nxt[0])]:
                #print(current, nxt, " Deadhead")
                path += [current, nxt]
                continue
        #handles last service in log
        last = log[-1]
        dum = (last[0], last[1] + 4)
        if (dum, nxt) in A_s:
            #print(current, nxt, " Service")
            path += [(last[0], last[1] + t) for t in range(1, 5)]
            path += [nxt]
        #now read the path
        fuel_level = self.carac_heli[h]["init_fuel"]
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
                    print(h, " starts service in ", prev[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60)), "and finishes in ", succ[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60)), " - Fuel level at arrival: ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), "%", " - Minimum level reached " * int(fuel_level < 325) )
                else:
                    print(h, " leaves from ", prev[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60)), "and arrives in ", succ[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60)), " - Fuel level at arrival: ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), "%", " - Minimum level reached " * int(fuel_level < 325))

            else:
                if a in A_g:
                    fuel_level = self.carac_heli[h]["fuel_cap"]
                    print(h, " starts refueling in ", prev[0], " at ",'{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60)), ", finishes at ", '{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60)), " - Fuel level at arrival: ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), "%", " - Minimum level reached " * int(fuel_level < 325))
        print("")
        #print(served, " requests are served out of a total of ", len(self.r))
        return path



    def fuel_checker(self, sol):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        ch, cp, cache_log = self.compute_cost(sol)
        fuel_viol = 0
        for h in ch:
            fuel_viol += ch[h][1]
        return fuel_viol == 0


    def neighbourhood_shake(self, x, neigh):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        if neigh not in ["refuel_neigh", "shift_neigh", "swap_paths_neigh", "swap_neigh"]:
            raise AttributeError(f"{neigh} is not an implemented neighborhood. Use one of refuel_neigh, shift_neigh, swap_paths_neigh, swap_neigh")
        N = getattr(self, neigh)
        if neigh in ["refuel_neigh", "swap_paths_neigh"]:
            xp, cost_xp = N(x, eps=1)
        elif neigh == "shift_neigh":
            unserved, served = self.update_served_status(x)
            xp, cost_xp = N(x, served, unserved, eps=1)
        elif neigh == "swap_neigh":
            unserved, served = self.update_served_status(x)
            xp, cost_xp = N(x, served, eps=1)

        return xp, cost_xp

    def neighbourhood_change(self, x, xp, cost_x, cost_xp, k):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        if cost_xp < cost_x:
            return xp, "No Change"
        else:
            return k+1, "Change"




    def VNS_module(self, sol, verbose=False, no_imp = 50, max_iter = 900):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        x = sol.copy()
        cost_heli, cost_pen, _ = self.compute_cost(x)
        cost_x = cost_pen
        for h in cost_heli:
            cost_x += cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]
        if verbose:
            print("")
            print("---------")
            print(f"Starting VNS module with cost : {cost_x} and fuel lambda : {self.pen_fuel}.")
        stag = 0
        neighbourhoods = {0:"shift_neigh", 1:"refuel_neigh", 2: "shift_neigh",
                          3:"swap_paths_neigh", 4:"shift_neigh", 5: "refuel_neigh",
                          6:"swap_neigh", 7:"refuel_neigh", 8:"shift_neigh", 9:"refuel_neigh"}
        k = 0
        it = 0
        while stag < no_imp and it < max_iter:
            it += 1
            #Shaking step
            xp, cost_xp = self.neighbourhood_shake(x, neighbourhoods[k])
            #First improvement
            N = getattr(self, neighbourhoods[k])
            if neighbourhoods[k]  in ["refuel_neigh", "swap_paths_neigh"]:
                xpp, cost_xpp = N(xp, eps=0)
            elif neighbourhoods[k] == "shift_neigh":
                unserved, served = self.update_served_status(xp)
                xpp, cost_xpp = N(xp, served, unserved, eps=0)
            elif neighbourhoods[k] == "swap_neigh":
                unserved, served = self.update_served_status(xp)
                xpp, cost_xpp = N(xp, served, eps=0)
            #Decide to change neighbourhoods or keep using the same one
            change = self.neighbourhood_change(x, xpp, cost_x, cost_xpp, k)
            if change[1] == "No Change":
                x = xpp
                cost_x = cost_xpp
                stag = 0
                if verbose:
                    print(f"Obtained new best solution with cost {cost_x} at iteration {it}. Current neighbourhood is {neighbourhoods[k]}")
            elif change[1] == "Change":
                stag += 1
                k = k + 1 if k < len(neighbourhoods) - 1 else 0
        if verbose :
            print("")
            print("Ending VNS module")
            print("---------")
            print("")
        return x, cost_x



    def MH2(self, init_sol, n_iter, rate_penalty, verbose=False, verbose2=False, no_imp_vns = 100, max_iter_vns = 1000):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        start = time.time()
        x = init_sol.copy()
        cost_heli, cost_pen, _ = self.compute_cost(x)
        cost_x = cost_pen
        violation_x = 0
        for h in cost_heli:
            cost_x += cost_heli[h][0]
            violation_x += self.pen_fuel * cost_heli[h][1]
        if verbose:
            print("---------")
            print(f"Starting MH2 with cost : {cost_x} and violation {violation_x} and fuel lambda : {self.pen_fuel}.")
            print("---------")
        lbdas = []
        distances = []
        hist_obj = []
        hist_viol = []
        hist_best = []
        stag = 0
        stag_in_viol = 0
        prev_violation = violation_x
        eps = 1
        best = x.copy()
        best_cost = cost_x
        for it in range(n_iter):
            xp, cost_xp = self.VNS_module(x, verbose=verbose2, no_imp = no_imp_vns, max_iter = max_iter_vns)
            #got local minima with current value of fuel penalty
            #----
            #updating fuel penalty lambda
            cost_heli_xp, cost_pen_xp, _ = self.compute_cost(xp)

            violation_xp = 0
            for h in cost_heli_xp:
                violation_xp += cost_heli_xp[h][1]

            rate_penalty = self.adap(hist_viol, hist_obj, lbdas, rate_penalty, cost_xp, violation_xp, eps, stag_in_viol)

            hist_viol.append(violation_xp)
            hist_obj.append(cost_xp)

            if prev_violation == violation_xp:
                stag_in_viol += 1

            prev_violation = violation_xp

            #update the lambda for fuel violation
            self.pen_fuel += rate_penalty * violation_xp

            lbdas.append(self.pen_fuel)
            distances.append(np.linalg.norm(xp - init_sol, ord=1))
            if cost_x <= cost_xp and violation_xp == 0:
                stag += 1
            x = xp
            cost_x = cost_xp
            if verbose and it % 10 == 0:
                print(f"Current cost is {cost_x} and current violation is {violation_xp}.")
                print(f"Current fuel penalty is {self.pen_fuel} .")
            if cost_x < best_cost and violation_xp == 0:
                best = x
                best_cost = cost_x
                hist_best.append(best_cost)
            if stag > 200:
                break
#             if stag_in_viol > 30 and hist_viol[-1] != 0:
#                 rate_penalty *= 1.1
            eps -= 0.01



        if verbose:
            duration = round(time.time() - start, 2)
            print(f"Mh2 terminated in {duration}. Number of move per second is {round(self.move / duration, 2)}")
        return best, best_cost, lbdas, distances, hist_viol, hist_obj, hist_best


    def adap(self, ho, hv, lbdas, rate, cost, viol, eps, stag_in_viol):
        if len(ho) == 0:
            return rate
        u = np.random.random()
        if stag_in_viol > 10 and viol > 0:
            return rate * 2
        if cost <= ho[-1] and viol <= hv[-1]:
            return rate
        elif cost <= ho[-1] and viol > hv[-1]:
            d = (viol - hv[-1]) / hv[-1]
            return rate * (1 + d)
        elif cost > ho[-1] and viol <= hv[-1]:# and u < eps:
            self.pen_fuel /= 2
            return rate / 2
        elif cost > ho[-1] and viol <= hv[-1]:# and u >= eps:
            return rate / 2
        return rate


    def restart_heuri(self, queue):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        mat = queue[0].reshape((1, len(queue[0])))
        for k in range(1, len(queue)):
            mat = np.concatenate((mat, queue[k].reshape((1, len(queue[0])))), axis=0)

        return np.product(mat, axis=0)


    def check_refuel_ins(self, seq, bit):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        h = bit // (2 * len(self.r))
        start = self.carac_heli[self.helicopters[h]]["start"]
        if seq[bit] == 0:
            return True
        if self.indices[bit] == "Refuel0":
            if not(np.any(seq[self.assign[self.helicopters[h]]][1:])):
                return False
            else:
                nxt = next(i for i in range(bit+1, len(seq)) if seq[i] == 1)
                if not((start, self.indices[nxt]) in self.refuel_compatible):
                    return False
        else:
            if seq[bit - 1] == 0:
                return False
            prev = bit - 1
            try:
                nxt = next(i for i in range(bit+1, len(seq)) if seq[i] == 1)
            except StopIteration:
                nxt = None

            if not(nxt) or nxt not in self.assign[self.helicopters[h]]:
                return False

            if not((self.indices[prev], self.indices[nxt]) in self.refuel_compatible):
                return False
        return True



    def check_insertion(self, seq, h, req):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        ind_req = self.reverse_indices[h][req]
        #print(ind_req)
        start = self.carac_heli[h]["start"]
        #print("Indice is ", ind_req)
        try:
            nxt = next(i for i in range(ind_req+1, len(seq)) if seq[i] == 1)
        except StopIteration:
            nxt = None
        try:
            prev = next(ind_req - i for i in range(1, ind_req+1) if seq[ind_req - i] == 1 and not ("Ref" in self.indices[ind_req - i]))
        except StopIteration:
            prev = None

        #print("Prev is ", prev, self.indices[prev], " Next is ", nxt, self.indices[nxt])
        if nxt and nxt in self.assign[h] and not((req, self.indices[nxt]) in self.time_compatible):
                return False
        if prev and prev in self.assign[h]:
            if seq[prev + 1] == 1 and not ((self.indices[prev], req) in self.refuel_compatible):
                return False
            if not ((self.indices[prev], req) in self.time_compatible):
                return False
        elif not(prev) or not(prev in self.assign[h]):

            if seq[self.reverse_indices[h]["Refuel0"]] == 1 and not ((start, req) in self.refuel_compatible):
                return False
            if not ((start, req) in self.time_compatible):
                return False
        return True

    def swap_paths_neigh(self, seq, eps=0):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        queue = []
        #best_cost, best = np.inf, seq
        cost_heli, cost_pen, _ = self.compute_cost(seq)
        #cost = sum(cost_heli.values()) + cost_pen
        cost = cost_pen
        for h in cost_heli:
            cost += cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]
        #cost of input point
        cost_current = cost

        for i in range(len(self.helicopters)):
            rolled_seq = np.roll(seq, 2*len(self.r)*i)

            cost_heli_rolled, cost_pen_rolled, _ = self.compute_cost(rolled_seq)
            cost_cand = cost_pen_rolled
            for h in cost_heli_rolled:
                #print(cost_heli_rolled[h])
                cost_cand += cost_heli_rolled[h][0] + self.pen_fuel * cost_heli_rolled[h][1]

            #--- First improvement is returned ---
            u = np.random.random()
            if cost_cand < cost_current or (u <= eps and cost_cand < np.inf):
                return rolled_seq, cost_cand

        return seq, cost_current




    def shift_neigh(self, seq, served, unserved, eps=0):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        cost_heli, cost_pen, _ = self.compute_cost(seq)
        #cost = sum(cost_heli.values()) + cost_pen
        cost = cost_pen
        for h in cost_heli:
            cost += cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]

        #ties broken randomly
        d_r = self.r.copy()
        l = list(d_r.items())
        np.random.shuffle(l)
        d_r = dict(l)
        #cost of input point
        cost_current = cost

        for req in d_r:
            if req in unserved:
                for h in self.helicopters:
                    self.move += 1
                    cand = self.add_request(seq, req, h)
                    feas = self.check_insertion(cand, h, req)
                    #feas = self.feasible(cand)
                    if not (feas):
                        continue

                    op_cand, viol_cand = self.compute_cost_heli(cand, h, log=False)
                    cand_cost_h = op_cand + self.pen_fuel*viol_cand#self.compute_cost_heli(cand, h, log=False)
                    cost_cand = cost - (cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]) + cand_cost_h - self.pen_unserved

                    #--- First improvement is returned ---
                    u = np.random.random()
                    if cost_cand < cost_current or (u <= eps and cost_cand < np.inf):
                        return cand, cost_cand


            else:
                self.move += 1
                cand, h = self.remove_request(seq, req)
                op_cand, viol_cand = self.compute_cost_heli(cand, h, log=False)
                cand_cost_h = op_cand + self.pen_fuel*viol_cand#self.compute_cost_heli(cand, h, log=False)

                cost_cand = cost - (cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]) + cand_cost_h + self.pen_unserved
                #print("Removing", req, " from ", h, " New cost heli", cand_cost_h, " New cost total = ", cost_cand)

                #--- First improvement is returned ---
                u = np.random.random()
                if cost_cand < cost_current or (u<=eps and cost_cand < np.inf):
                    return cand, cost_cand


                for hb in self.helicopters:
                    self.move += 1
                    if hb != h:
                        candu = self.add_request(cand, req, hb)

                        #feas = self.feasible(candu)
                        feas = self.check_insertion(candu, hb, req)
                        if not(feas):
                            continue
                        op_candu, viol_candu = self.compute_cost_heli(candu, hb, log=False)
                        cand_cost_hb = op_candu + self.pen_fuel*viol_candu#self.compute_cost_heli(candu, hb, log=False)
                        cost_candu = cost_cand - (cost_heli[hb][0] + self.pen_fuel * cost_heli[hb][1]) + cand_cost_hb - self.pen_unserved
                        #print("Previous cost cand : ", cost_cand, " old cost heli ", cand_cost_h)
                        #print("Adding", req, " to ", hb, " New cost heli", cand_cost_hb, " New cost total = ", cost_candu)
                        #--- First improvement is returned ---
                        u = np.random.random()
                        if cost_candu < cost_current or (u <= eps and cost_candu < np.inf):
                            return candu, cost_candu

        #if no better neighbor was found, return input point and cost
        return seq, cost_current


    def swap_neigh(self, seq, served, eps=0):
        cost_heli, cost_pen, _ = self.compute_cost(seq)
        #cost = sum(cost_heli.values()) + cost_pen
        cost = cost_pen
        for h in cost_heli:
            cost += cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]

        #ties broken randomly
        d_r = served.copy()
        np.random.shuffle(d_r)
        #best, best_cost = seq, cost
        served_by = {}
        #storing cost of input point
        cost_current = cost
        for r in d_r:
            served_by[r] = [i for i in range(len(self.helicopters)) if seq[self.reverse_indices["h1"][r] + 2 * len(self.r) * i] == 1][0]
        for k in range(len(served)):
            for l in range(k):
                if k == l:
                    continue
                r1 = d_r[k]
                r2 = d_r[l]

                if served_by[r1] == served_by[r2]:
                    continue
                self.move += 4
                #swapping
                cand, h1 = self.remove_request(seq, r1)
                cand, h2 = self.remove_request(cand, r2)
                cand = self.add_request(cand, r2, h1)
                cand = self.add_request(cand, r1, h2)

                feas = self.check_insertion(cand, h1, r2) and self.check_insertion(cand, h2, r1)

                if not(feas):
                    continue
                op_cand1, viol_cand1 = self.compute_cost_heli(cand, h1, log=False)
                cand1_cost_h = op_cand1 + self.pen_fuel * viol_cand1#self.compute_cost_heli(cand, h1, log=False)
                op_cand2, viol_cand2 = self.compute_cost_heli(cand, h2, log=False)
                cand2_cost_h = op_cand2 + self.pen_fuel * viol_cand2#self.compute_cost_heli(cand, h2, log=False)

                cost_cand = cost - (cost_heli[h1][0] + self.pen_fuel * cost_heli[h1][1] + cost_heli[h2][0] + self.pen_fuel * cost_heli[h2][1]) + cand1_cost_h + cand2_cost_h

                #--- First improvement is returned ---
                u = np.random.random()
                if cost_cand < cost_current or (u <= eps and cost_cand < np.inf):
                    return cand, cost_cand

        #if no better neighbor was found, return input point and cost
        return seq, cost_current

    def refuel_neigh(self, seq, eps=0):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """

        cost_heli, cost_pen, _ = self.compute_cost(seq)
        cost = cost_pen
        for h in cost_heli:
            cost += cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]
        cost_current = cost #cost of input solution, i.e. x
        idxrf = list(self.ind_type_seq["Refuel"])
        np.random.shuffle(idxrf)
        for rf in idxrf:
            h = rf // (2 * len(self.r))
            if h == (rf - 1) // (2 * len(self.r)) and seq[rf - 1] == 0:
                continue
            self.move += 1
            cand = seq.copy()
            cand[rf] = 1 - cand[rf]
            feas = self.check_refuel_ins(cand, rf)
            if not(feas):
                continue
            h = rf // (2 * len(self.r))
            h = self.helicopters[h]
            op_cand, viol_cand = self.compute_cost_heli(cand, h, log=False)
            cost_cand_heli = op_cand + self.pen_fuel * viol_cand#self.compute_cost_heli(cand, h, log=False)
            cost_cand = cost - (cost_heli[h][0] + self.pen_fuel * cost_heli[h][1]) + cost_cand_heli

            #--- First improvement is returned ---
            u = np.random.random()
            if cost_cand < cost_current or (u <= eps and cost_cand < np.inf):
                return cand, cost_cand

        #if no better neighbor was found, return input point and cost
        return seq, cost_current




    def add_request(self, seq, req, h):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """

        idrh = self.reverse_indices[h][req] #[i for i in idh if req == self.indices[i]][0]
        seq_copy = seq.copy()
        seq_copy[idrh] = 1
        return seq_copy

    def remove_request(self, seq, req):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        seq_copy = seq.copy()
        idrs = list(np.where(seq_copy==1)[0])
        idr = [i for i in idrs if req == self.indices[i]][0]
        h = idr // (2*len(self.r))

        seq_copy[idr] = 0
        #remove the refuels that have become inconsistent with the structure due to this removal
        if idr+1 in self.assign[self.helicopters[h]]:
          seq_copy[idr + 1] = 0  #remove the next refuel if it was there
        #remove first refuel of the chain if there only one bit set to one.
        if np.sum(seq_copy[self.assign[self.helicopters[h]]]) == 1:
          fr = self.assign[self.helicopters[h]][0]
          seq_copy[fr] = 0
        #remove refuel that was just before if req was the last request served
        ones = np.where(seq_copy == 1)[0]
        ones = list(set(ones).intersection(set(self.assign[self.helicopters[h]])))
        ones.sort()
        if len(ones) > 0:
            if "Ref" in self.indices[ones[-1]]:
                seq_copy[ones[-1]] = 0

        return seq_copy, self.helicopters[h]





    def update_served_status(self, seq):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        idx = np.where(seq == 1)[0]
        served = [self.indices[i] for i in idx if not("Ref" in self.indices[i])]
        unserved = set(self.r.keys()) - set(served)
        return list(unserved), list(served)

    def init_heuristic(self):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        #print(hasattr(self, "empty_sol"))
        if not(hasattr(self, "empty_sol")):
            print("dd")
            raise AttributeError("self has not attribute empty_sol. Call init_encoding before using this method.")

        served = []
        init_sol = self.empty_sol.copy()
        #cache = []
        for h in self.assign:
            #seq = init_sol[assign[h]]
            routeh = []
            start = self.carac_heli[h]["start"]
            for i in self.assign[h]:
                if routeh == [] and (start, self.indices[i]) in self.time_compatible and not(self.indices[i] in served):
                    init_sol[i] = 1
                    served.append(self.indices[i])
                    routeh.append(self.indices[i])
                    #self.service_heli[h].append(self.request_id[self.indices[i]])
                elif routeh and (routeh[-1], self.indices[i]) in self.time_compatible and not(self.indices[i] in served):
                    init_sol[i] = 1
                    served.append(self.indices[i])
                    routeh.append(self.indices[i])
                    #self.service_heli[h].append(self.request_id[self.indices[i]])
                #cache.append(init_sol.copy())

        return init_sol  #, cache

    def init_heuristic_random(self):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        served = []
        init_sol = self.empty_sol.copy()
        served_heli = {h:[] for h in self.helicopters}
        served = []
        n = 2*len(self.r)
        #cache = []
        o = self.ind_type_heli["h1"]["Request"].copy()
        np.random.shuffle(o)
        for i in o:
            if i in served:
                continue
            #get feasible helis
            cand_h = []
            for h in self.helicopters:
                start = self.carac_heli[h]["start"]
                if served_heli[h] == []:
                    if (start, self.indices[i]) in self.time_compatible:
                        cand_h.append(h)
                else:
                    if (served_heli[h][-1], self.indices[i]) in self.time_compatible:
                        cand_h.append(h)
            #choose random assignment - uniformly
            if not(cand_h):
                #no more insertion possible
                break
            h_elected = np.random.choice(cand_h)
            heli = self.helicopters.index(h_elected)
            ind = i + heli * n
            init_sol[ind]=1
            served_heli[h_elected].append(self.indices[i])
            served.append(i)
        return init_sol

    def compute_imbalance(self, sol):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

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
        #print(mu, std)
        #if len(nb_s) == 2:
        #  return abs(nb_s[0] - nb_s[1])
        if mu == 0 or std == 0:
            return 0
        return std / mu

    def viz_graph(self, arcs, A, A_s, A_g, A_f, colors, note=""):
        """ Viz functions for the time expanded network, schedule is represented on graph.
          --------------
          Params :
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
        img=mpimg.imread('image/icon.png')
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
        plt.title("Graph Schedule, Rand." + str(len(self.r)) + ".H" + str(len(self.helicopters)) + note)
        #nx.draw(G, pos, alpha=0.2, node_size=5, ax=ax, node_color='black')
        nx.draw_networkx_nodes(G, pos, node_size=5, ax=ax, node_color='black')
        for h in self.helicopters:
            if h in arcs:
                nx.draw_networkx_edges(G,pos,
                                      edgelist=[e for e in A if e in arcs[h] and not(e in A_f)],
                                      width=4, alpha=1, edge_color=colors[h])
                nx.draw_networkx_edges(G,pos,
                                      edgelist=[e for e in A if e in arcs[h] and e in A_f],
                                      width=1, alpha=1, edge_color=colors[h], style="dotted")

                #nx.draw_networkx_edges(G,pos,
                #                    edgelist=[e for e in A_g if e in arcs[h]],
                #                    width=4, alpha=1, edge_color=colors[h])

        nx.draw_networkx_edges(G,pos,
                            edgelist=[e for e in A_s],
                            width=1, alpha=0.5, edge_color='g')

        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlabel("Time")
        plt.yticks([i for i in range(len(self.locations))], self.locations)
        #plt.xticks([i for i in range(len(self.T))][::15], ['{:02d}:{:02d}'.format(*divmod(870 + i, 60)) for i in range(len(self.T))][::15])
        #MVP
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
            #a.patch.set_edgecolor('black')
            #a.patch.set_linewidth('1')

        #plt.show()
        return fig


    def descibe_sol(self, sol, A_s, A_g, A, notes=""):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        print(notes)
        unserved, served = self.update_served_status(sol)
        cost = 0
        paths_sol = {}
        for h in self.helicopters:
            print("")
            costh, log = self.compute_cost_heli(sol, h)
            path = self.read_log(log, h, A_s, A_g, A)
            paths_sol[h] = path
            cost += costh[0] + self.pen_fuel * costh[1]
            print("---------")
        print("")
        print("Operationnal cost :", cost, " - Penalty for unserved demands : ", self.pen_unserved * len(unserved), " - Total :", cost + self.pen_unserved * len(unserved))
        print(len(served), " requests are served out of ", len(self.r))
        return paths_sol

    def get_arcs(self, paths_sol):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        arcs = {}
        for h in paths_sol:
            ar = []
            for i in range(1, len(paths_sol[h])):
                ar.append((paths_sol[h][i-1], paths_sol[h][i]))
            arcs[h] = ar
        return arcs

    def viz_convergence(self, cache_cost_best, cache_cost_current, notes=""):
        """ Desc
            --------------
            Params :
                    None
            --------------
            Returns :
                    None

        """
        plt.figure(figsize=(14,7))
        plt.plot(cache_cost_best, label="Best costs")
        plt.plot(cache_cost_current, label="Explored costs")
        plt.xlabel("Iterations")
        plt.title("Cost evolution - R" + str(len(self.r)) + ". H" + str(len(self.helicopters)) + notes)
        plt.legend()
        plt.savefig("image/vns_convergence.png")

