
import numpy as np
import itertools
import utils as U
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import optim
#from typeguard import typechecked
#--- Utils function for experiments ---

#@typechecked
def delay_request(
                  obj: optim.META_HEURISTIC,
                  req: Tuple[str, str, int],
                  sol: np.ndarray) -> Tuple[np.ndarray,
                                            Tuple[str, str, int]]:
  """ This function will simulate a delay in a request. Will be used to assess impact of delays on computed schedule.
    Req is going to be delayed by enough time so that solution is impacted. Then request is removed from the set of served in sol
    so that solution has to be re-opt.
    --------------
    Args :
            obj: Object instance of META_HEURISTIC Class in src/optim.py
            req: tuple, request to delay
            sol: np.ndarray, encoding of valid solution

    --------------
    Returns :
            new_sol: new solution encoding after delay, removing the service of req.
            new_req: request req updated with new departure time to reflect delay

  """

  ref = False
  hr = None
  for h in obj.helicopters:
    if sol[obj.reverse_indices[h][req]] == 1:
      hr = h
      if obj.reverse_indices[h][req] + 1 < len(sol) and sol[obj.reverse_indices[h][req] + 1] == 1:
        ref = True
      break

  try:
      nxt = next(i for i in range(obj.reverse_indices[hr][req]+2, len(sol)) if sol[i] == 1 and i in obj.assign[hr])
  except StopIteration:
      nxt = None

  if nxt is None:
    slack = 720 - obj.r[req][-1][1][1]
    delay = 2

  else:
    slack = obj.indices[nxt][2] - obj.r[req][-1][1][1] - int(ref) * obj.refuel_time[req[1]]
      #will delay req by delay to produce a re-opt
    delay = slack + 2
  new_req = {}
  for ro in obj.r:
    if ro == req:
      new_req[(req[0], req[1], req[2] + delay)] = [((ele[0][0], ele[0][1] + delay), (ele[1][0], ele[1][1] + delay)) for ele in obj.r[ro]]
    else:
      new_req[ro] = obj.r[ro]
  #now update sol
  new_sol, _ = obj.remove_request(sol, req)
  return new_sol, new_req

#@typechecked
def simulate_data(
                  l: int,
                  d: int,
                  h: int,
                  seed: int) -> Tuple[List[str],
                                      List[str],
                                      Dict[str, int],
                                      Dict[str, int],
                                      List[int],
                                      Dict[Tuple[str, str], int],
                                      Dict[str, int],
                                      Dict[str, int],
                                      int,
                                      int,
                                      int,
                                      int,
                                      int,
                                      List[Tuple[str, int]],
                                      Dict[str, Union[str, float, int]],
                                      Dict[Tuple[str, str, int], List[Tuple[str, int]]],
                                      List[Tuple[Tuple[str, int], Tuple[str, int]]],
                                      List[Tuple[Tuple[str, int], Tuple[str, int]]],
                                      List[Tuple[Tuple[str, int], Tuple[str, int]]],
                                      List[Tuple[Tuple[str, int], Tuple[str, int]]],
                                      List[Tuple[Tuple[str, int], Tuple[str, int]]]]:
    """ Simulates problems parameters and flights to serve. This serve to create the data to evaluate our
        optim algo. Eventually should sync with elevate to use more realistic data.
        Will create :
            - h helicopters with :  Starting skyport (random)
                                    Fuel tank capacity (units, 1100)
                                    Fuel consumption (units/minute, 20)
                                    Cost per minute ($34)
                                    Starting fuel level (900, 1000 or 1100)
            - l skyports with :     Landing fee ($) (100, 200 or 300)
                                    Parking fee ($) (0, 100, 200 or 300)
                                    Refuel price ($) (25: $100, 15: $200, 5: $700)
                                    Refuel time (minute) (5, 15 or 25)
                                    Fly time (minute) (5, 7, 10 or 15 minutes)
            - Requests and other problems params.
            --------------
            Args :
                    l: int, number of skyports to put in the infrastructure
                    d: int, number of demand to generate (flights)
                    h: int, number of helicopter to generate in the fleet
                    seed: int, seed for randomness
            --------------
            Returns :
                    helicopters: list, contains helicopter id
                    skyports: list, contains all skyports id
                    refuel_times: dict, contains time to refuel in minutes at each skyports
                    refuel_prices: dict, contains $ price of refueling at each skyports
                    T: list, contains all time steps
                    fly_time: dict, time to fly between skyports in minutes
                    fees: dict, price in $ to land at each locations
                    parking_fee: dict, price of parking fee in $ at each locations
                    beta: int, gain in $ of serving a request
                    min_takeoff: int, minimum quantity of fuel needed to takeoff
                    n_iter: int, number of iterations in the VNS
                    pen_fuel: int, initial fuel penalty
                    no_imp_limit: int, number of stagnation step before breaking VNS
                    nodes: list, contains all timed node of TEN
                    carac_heli: dict, contains characteristics for each heli
                    r: dict, contains all requests simulated
                    A: list, contains all arcs of TEN
                    A_f: list, contains all deadhead arcs
                    A_g: list, contains all refueling arcs
                    A_s: list, contains all service arcs
                    A_w: list, contains all waiting arcs
    """
    np.random.seed(seed)
    helicopters = [f"h{i}" for i in range(1, h+1)]
    skyports = [f"S{i}" for i in range(1, l + 1)]
    refuel_times = {loc: np.random.choice([25, 15, 5], p=np.array([0.5, 0.3, 0.2])) for loc in skyports}
    prices = {25: 100, 15: 200, 5: 700}
    refuel_prices = {loc: prices[refuel_times[loc]] for loc in skyports}
    T = [k for k in range(720)]  # timesteps

    # flying time between skyports
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


    n_request = d
    # Simulate requests
    r = U.simulate_requests(n_request, fly_time, skyports, T, verbose=0)
    A_w, A_s, A_f, A_g = U.create_arcs(nodes, r, fly_time, refuel_times)
    A = A_w + A_s + A_f + A_g

    return helicopters, skyports, refuel_times, refuel_prices, T, fly_time, fees, parking_fee, beta, min_takeoff, n_iter, pen_fuel, no_imp_limit, nodes, carac_heli, r, A, A_f, A_g, A_s, A_w


#@typechecked
def sim_failure(
                obj: optim.META_HEURISTIC,
                h: str,
                time: int,
                prev_sol: np.ndarray) -> Tuple[np.ndarray, Any, Any]:
    """ Simulate failure of aircraft h at time time. This helicopter will be useable after this time. This function
        helps simulate this failure by cutting the existing solution (prev_sol) at the time of failure. The schedule
        before failure is kept aside and frozen. And the remaining of the solution is adapted.
        --------------
        Args :
                h : str, helicopter ID
                time : int, value between 0 and 720
                prev_sol : numpy array, encode existing routing
        --------------
        Returns :
                 - updated sol to be re-opt
                 - frozen parts of the solution, if relevant
                 - new_helicopters, if relevant
    """
    # -- determine first bit impacted by failure and remove affected operations
    print(f"Simulating failure of helicopter {h} at {'{:02d}:{:02d}'.format(*divmod(420 + time, 60))}. This helicopter cannot be used after this date.")
    frozen_bits = []
    failure_req = None
    new_sol = prev_sol.copy()
    print("Determining impacted parts of existing schedule...")
    for i in obj.assign[h]:
        if "Ref" in obj.indices[i]:
            if failure_req is not None:
                new_sol[i] = 0
            continue
        else:
            req = obj.indices[i]
            if req[2] > time and failure_req is None:
                failure_req = req
                new_sol[i] = 0
                ind_req = i
            if failure_req is not None:
                new_sol[i] = 0
    frozen_bits += obj.assign[h]
    try:
        prev = next(ind_req - i for i in range(1, ind_req+1) if prev_sol[ind_req - i] == 1)
    except StopIteration:
        prev = None
    if prev is not None and "Ref" in obj.indices[prev]:
        new_sol[prev] = 0
    # -- check if failure will indeed have an impact
    if failure_req is None:
        print("Failure has no impact on current schedule.")
        return prev_sol, None, None
    #
    # -- Get new aircraft available for optim
    new_helicopters = list(set(obj.helicopters) - set([h]))
    # -- get set of bits that will have to be frozen when re-optimization happens

    for hb in new_helicopters:
        passed_time = False
        for i in obj.assign[hb]:
            if "Ref" in obj.indices[i]:
                continue
            else:
                if obj.indices[i][2] >= time and not(passed_time):
                    frozen_bits.append(i)
                    passed_time = True
                if obj.indices[i][2] <= time:
                    frozen_bits.append(i)
    return new_sol, frozen_bits, new_helicopters

