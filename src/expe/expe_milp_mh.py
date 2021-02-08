import sys
sys.path.append('/project/src/')
import numpy as np
import itertools
import operator
import matplotlib.pyplot as plt
import networkx as nx
import time
import utils as U
import matplotlib.image as mpimg
import copy
import heapq
import optim
import sys
import os
import click
import tempfile
from os import dup, dup2, close


# Read params

@click.command()
@click.argument("i", type=int)
@click.argument("l", type=int)
@click.argument("d", type=int)
@click.argument("h", type=int)
@click.argument("seed", type=int)
def main(i, l, d, h, seed):
    """ Will run MILP and MH on same pb and record metrics. the goal of this expe
        is to evaluate the performance of the MH.


        Args:
            i: int, id of experiement
            l: int, number of skyports
            d: int, number of demands (i.e. flights)
            h: int, number of helictopers in the fleet
            seed: int, for randomness

        Returns:
            None

    """
    print(f"Starting expe with params {i, l, d, h, seed}...")
    np.random.seed(seed)
    #helicopters = ["h1", "h2", "h3", "h4", "h5", "h6", "h7"]
    helicopters = [f"h{i}" for i in range(1, h+1)]
    #skyports = ["JFK", "JRB", "EWR", "Kearny", "S1", "S2", "S3", "S4"]  # N
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
    max_runtime = 2000
    no_imp_limit = 1600
    # nodes for the time space network

    header = [
        "Obj MH",
        "Valid MH",
        "Served MH",
        "Imb MH",
        "Time MH",
        "Obj MILP Time MH",
        "Status MILP Time MH",
        "Served MILP Time MH",
        "Imb MILP Time MH",
        "Obj MILP",
        "Status MILP",
        "Served MILP",
        "Imb MILP",
        "Time MILP"]
    res = np.array(header).reshape((1, len(header)))

    # ----------- Problem parameters ----------

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

    # sink nodes for helicopters
    sinks = {h: ("sink %s" % h, max(T)) for h in helicopters}

    # Simulate requests
    r = U.simulate_requests(n_request, fly_time, skyports, T, verbose=0)

    A_w, A_s, A_f, A_g = U.create_arcs(nodes, r, fly_time, refuel_times)

    A = A_w + A_s + A_f + A_g

    # ------ Solving using MH -------
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

    meta.init_encoding()
    meta.init_compatibility()
    meta.init_request_cost()

    init_heuri = meta.init_heuristic()
    best_sol, best_cost, _, _, perf_over_time, _, _ = meta.VNS(
        init_heuri, n_iter, eps=0.12, no_imp_limit=no_imp_limit, random_restart=200, verbose=0)
    fuel_viol = meta.fuel_checker(best_sol)
    dur = round(time.time() - start, 3)
    imb_mh = meta.compute_imbalance(best_sol)
    _, served = meta.update_served_status(best_sol)
    print("MH done. Starting MILP on time mh...")

    # ------ Using MILP timed out at MH level --------
    schedule_opt2 = optim.MILP2(
        helicopters,
        nodes,
        r,
        A,
        skyports,
        T,
        carac_heli,
        sinks,
        parking_fee,
        A_w,
        A_s,
        A_f,
        A_g,
        beta,
        fees,
        fly_time,
        min_takeoff,
        refuel_prices,
        h,
        refuel_times,
        meta.pen_unserved)
    schedule_opt2.build_model()



    with tempfile.TemporaryFile() as tmp_output:
      orig_std_out = dup(1)
      dup2(tmp_output.fileno(), 1)

      status_time_mh = schedule_opt2.solve(
        max_time=dur,
        opt_gap=0.0,
        verbose=1,
        verbose2=1)
      dup2(orig_std_out, 1)
      close(orig_std_out)
      tmp_output.seek(0)
      logs = tmp_output.read().splitlines()
      for ele in logs:
        ele = ele.decode("utf-8")
        if ("Result" in ele):
          status_milp_time_mh = ele


    _, nb_uns_time_mh = schedule_opt2.convert_sol()
    imb_milp_time_mh = schedule_opt2.get_imbalance()
    obj_milp_time_mh = status_time_mh[1]
    #time_m_time_mh = status_time_mh[2]
    print("MILP done. Starting MILP on maxruntime...")

    # ------- Solving using MILP 2 -------
    schedule_opt2 = optim.MILP2(
        helicopters,
        nodes,
        r,
        A,
        skyports,
        T,
        carac_heli,
        sinks,
        parking_fee,
        A_w,
        A_s,
        A_f,
        A_g,
        beta,
        fees,
        fly_time,
        min_takeoff,
        refuel_prices,
        h,
        refuel_times,
        meta.pen_unserved)
    schedule_opt2.build_model()



    with tempfile.TemporaryFile() as tmp_output:
      orig_std_out = dup(1)
      dup2(tmp_output.fileno(), 1)

      status = schedule_opt2.solve(
        max_time=max_runtime,
        opt_gap=0.0,
        verbose=1,
        verbose2=1)
      dup2(orig_std_out, 1)
      close(orig_std_out)
      tmp_output.seek(0)
      logs = tmp_output.read().splitlines()
      for ele in logs:
        ele = ele.decode("utf-8")
        if ("Result" in ele):
          status_milp = ele


    _, nb_uns = schedule_opt2.convert_sol()
    imb_milp = schedule_opt2.get_imbalance()
    obj_milp = status[1]
    time_m = status[2]
    print("Caching results...")
    row = [
        best_cost,
        fuel_viol,
        len(served),
        imb_mh,
        dur,
        obj_milp_time_mh,
        "Optimal" in status_milp_time_mh,
        d - nb_uns_time_mh,
        imb_milp_time_mh,
        obj_milp,
        "Optimal" in status_milp,
        d - nb_uns,
        imb_milp,
        time_m]


    res = np.concatenate((res, np.array(row).reshape(1, len(header))), axis=0)

    expe_name = f"/tmp/results/Expe_Params_{i}_{l}_{d}_{h}_{seed}_Eps_{0.12}_StopStag_{no_imp_limit}__MHNoRestart_MaxTimeMilp_{max_runtime}__RandDemand_RandStartHeli_TimeDay0700_1900.txt"
    expe_name_perf = f"/tmp/results/Expe_Params_{i}_{l}_{d}_{h}_{seed}_Perf_Eps_{0.12}_StopStag_{no_imp_limit}__MHNoRestart_MaxTimeMilp_{max_runtime}__RandDemand_RandStartHeli_TimeDay0700_1900.txt"

    os.makedirs("/tmp/results/", exist_ok=True)

    np.savetxt(expe_name_perf, np.array(perf_over_time), fmt="%s", delimiter=",")
    np.savetxt(expe_name, res, fmt="%s", delimiter=",")

    np.savetxt("/tmp/results/DONE", np.array([]))
    print("Done")


if __name__ == "__main__":
    main()
