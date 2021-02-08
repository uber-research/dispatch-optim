import sys
sys.path.append('/project/src/')
import numpy as np
import itertools
import operator
import time
import utils as U
import copy
import heapq
import optim
import os
import click
import utils_expe as ue


# Read params

@click.command()
@click.argument("i", type=int)
@click.argument("l", type=int)
@click.argument("d", type=int)
@click.argument("h", type=int)
@click.argument("seed", type=int)
@click.argument("seed_per", type=int)
def main(i, l, d, h, seed, seed_per):
    """ The goal of this expe is to see the gain of restarting the MH from a previously pertubed solution
        rather than starting from scratch. Measuring how far we end up in both cases.

        Args:
            i: int, id of expe
            l: int, number of skyports
            d: int, number of demands (flights)
            h: int, number of heli in the fleet
            seed: int, seed for infrastructure simulation
            seed_per: int, seed for random perturbation in the demands.


    """
    print(f"Starting current expe with {i, l, d, h, seed, seed_per}...")
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
        fly_time[(s1, s2)] = np.random.choice([5, 7, 10, 15])
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
    header = [
        "Obj MH",
        "Valid MH",
        "Served MH",
        "Imb MH",
        "Time MH",
        "Obj Adapt MH",
        "Status Adapt MH",
        "Served Adapt MH",
        "Imb Adapt MH",
        "L1 Adapt to Init",
        "Time Adapt MH",
        "Obj Restart MH",
        "Status Restart MH",
        "Served Restart MH",
        "Imb Restart MH",
        "Time Restart MH",
        "L1 Restart to Init"] #time adapt mh and L1 adapt mh are inverted.
    res = np.array(header).reshape((1, len(header)))

    print("Data simulated")
    #---- Run Mh a first time to get solutions
    start = time.time()
    meta = optim.META_HEURISTIC(
        r,
        helicopters,
        carac_heli,
        refuel_times,
        refuel_prices,
        skyports,
        nodes,
        parking_fee,
        fees,
        fly_time,
        T,
        beta,
        min_takeoff,
        pen_fuel,
        A_s,
        A,
        A_g)

    meta.init_compatibility()
    meta.init_request_cost()
    meta.init_encoding()
    init_heuri = meta.init_heuristic_random()
    best_sol, best_cost, _, _, perf_over_time, _, _ = meta.VNS(
        init_heuri, n_iter, eps=0.12, no_imp_limit=no_imp_limit, random_restart=200, verbose=0)
    fuel_viol = meta.fuel_checker(best_sol)
    dur = round(time.time() - start, 3)
    imb_mh = meta.compute_imbalance(best_sol)
    _, served = meta.update_served_status(best_sol)

    row = [best_cost, int(fuel_viol), len(served) / d, imb_mh, dur]
    #meta.descibe_sol(best_sol, A_s, A_g, A)
    print("MH finished. Pertubing flights....")
    # ---  Apply pertubation to one random requests
    np.random.seed(seed_per)

    ir = np.random.choice(range(len(served)))
    req = served[ir]
    print(f"{req} is going to be delayed..")
    per_sol, new_req = ue.delay_request(meta, req, best_sol)

    meta.r = new_req
    meta.init_encoding()
    meta.init_compatibility()
    meta.init_request_cost()

    A_w, A_s, A_f, A_g = U.create_arcs(meta.nodes, meta.r, meta.fly_time, meta.refuel_time)

    A = A_w + A_s + A_f + A_g
    #start from previous solution ..
    print("Starting to re-opt from previous sol...")
    start = time.time()
    ad_sol, ad_cost, _, _, ad_perf_over_time, _, _ = meta.VNS(
        per_sol, n_iter, eps=0.12, no_imp_limit=no_imp_limit, random_restart=200, verbose=0)
    ad_fuel_viol = meta.fuel_checker(ad_sol)
    ad_dur = round(time.time() - start, 3)
    ad_imb_mh = meta.compute_imbalance(ad_sol)
    _, ad_served = meta.update_served_status(ad_sol)
    #meta.descibe_sol(ad_sol, A_s, A_g, A)
    row += [ad_cost, int(ad_fuel_viol), len(ad_served) / d, ad_imb_mh, ad_dur, np.linalg.norm(ad_sol - best_sol, ord=1)]
    #start again from scratch ...
    print("Starting from scratch...")
    init_heuri = meta.init_heuristic_random()
    start = time.time()
    s_sol, s_cost, _, _, s_perf_over_time, _, _ = meta.VNS(
        init_heuri, n_iter, eps=0.12, no_imp_limit=no_imp_limit, random_restart=200, verbose=0)
    s_fuel_viol = meta.fuel_checker(s_sol)
    s_dur = round(time.time() - start, 3)
    s_imb_mh = meta.compute_imbalance(s_sol)
    _, s_served = meta.update_served_status(s_sol)
    #---
    #meta.descibe_sol(s_sol, A_s, A_g, A)
    row += [s_cost, int(s_fuel_viol), len(s_served) / d, s_imb_mh, s_dur, np.linalg.norm(s_sol - best_sol, ord=1)]
    #print(row)
    #--caching all
    res = np.concatenate((res, np.array(row).reshape(1, len(header))), axis=0)
    print("Finished current expe. Caching results...")
    #expe_name = f"/tmp/results/Expe_Params_VAR_{i}_{l}_{d}_{h}_{seed}_{seed_mh}_Eps_{0.12}_StopStag_{no_imp_limit}__MHNoRestart_MaxTimeMilp_{max_runtime}__RandDemand_RandStartHeli_TimeDay0700_1900.txt"
    expe_name = f"/tmp/results/Expe_Params_PertubMH_RandStart_{i}_{l}_{d}_{h}_{seed}_{seed_per}_Eps_{0.12}_StopStag_{no_imp_limit}__MHNoRestart_RandDemand_RandStartHeli_TimeDay0700_1900.txt"

    os.makedirs("/tmp/results/", exist_ok=True)
    np.savetxt(expe_name, res, fmt="%s", delimiter=",")

    np.savetxt("/tmp/results/DONE", np.array([]))
    print("Done.")




if __name__ == "__main__":
    main()