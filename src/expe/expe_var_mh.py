import sys
sys.path.append('/project/src/')
import numpy as np
import itertools
import operator
import matplotlib.pyplot as plt
import time
import utils as U
import matplotlib.image as mpimg
import copy
import heapq
import optim
import click
import os
import cProfile
import utils_expe as ue



@click.command()
@click.argument("i", type=int)
@click.argument("l", type=int)
@click.argument("d", type=int)
@click.argument("h", type=int)
@click.argument("seed", type=int)
@click.argument("seed_mh", type=int)
def main(i, l, d, h, seed, seed_mh):
    """ This expe is to evaluate the variance of the MH.

        Args:
            i: int, id of expe
            l: int, number of skyports to simulate
            d: int, number of demands to simulate
            h: int, number of heli in the fleet
            seed: int, seed for infra generation
            seed_mh: int, seed for MH

        Returns:
            None


    """
    print(f"Starting expe : Variance for MH with params {i, l, d, h, seed, seed_mh}")
    # -- step 0 simulate infrastructure
    helicopters, skyports, refuel_times, refuel_prices, T, fly_time, fees, parking_fee, beta, min_takeoff, n_iter, pen_fuel, no_imp_limit, nodes, carac_heli, r,  A, A_f, A_g, A_s, A_w = ue.simulate_data(l, d, h, seed)
    print("Data simulated...")


    header = [
        "Obj MH",
        "Operationnal Cost",
        "Deadhead Cost",
        "Refuel Cost",
        "Valid MH",
        "Served MH",
        "Imb MH",
        "Time MH"]
    res = np.array(header).reshape((1, len(header)))

    print("Starting MH....")
    print("")
    # ------ Solving using MH -------
    start = time.time()
    np.random.seed(seed_mh) # setting different see for mh execution
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

    #Following line changes if I run the Mh with a random init sol or with a deterministic one.
    #init_heuri = meta.init_heuristic()
    init_heuri = meta.init_heuristic_random()
    best_sol, best_cost, _, _, perf_over_time, dist, logs = meta.VNS(
        init_heuri, n_iter*2, eps=0.12, no_imp_limit=no_imp_limit, random_restart=200, verbose=1)
    fuel_check = meta.fuel_checker(best_sol)
    print(f"Is valid : {fuel_check}")
    print(meta.compute_cost(best_sol)[0])
    dur = round(time.time() - start, 3)
    imb_mh = meta.compute_imbalance(best_sol)
    unserved, served = meta.update_served_status(best_sol)
    paths_sol = meta.descibe_sol(best_sol, A_s, A_g, A, notes="", verbose=0)
    bd_cost = meta.cost_breakdown(best_sol, paths_sol, A, A_s, A_g)
    op_cost = best_cost - meta.pen_unserved * len(unserved)
    row = [
    best_cost,
    op_cost,
    bd_cost["cost_dh"],
    bd_cost["cost_ref"],
    int(fuel_check),
    len(served) / d,
    imb_mh,
    dur
    ]
    res = np.concatenate((res, np.array(row).reshape(1, len(header))), axis=0)
    print("Finished current expe. Caching results...")
    #expe_name = f"/tmp/results/Expe_Params_VAR_{i}_{l}_{d}_{h}_{seed}_{seed_mh}_Eps_{0.12}_StopStag_{no_imp_limit}__MHNoRestart_MaxTimeMilp_{max_runtime}__RandDemand_RandStartHeli_TimeDay0700_1900.txt"
    expe_name = f"/tmp/results/Expe_Params_VAR_MVP_{i}_{l}_{d}_{h}_{seed}_{seed_mh}_Eps_{0.12}_StopStag_{no_imp_limit}__MHNoRestart_RandDemand_RandStartHeli_TimeDay0700_1900.txt"
    print(row)
    expe_name_perf = f"/tmp/results/Expe_Params_VAR_MVP_{i}_{l}_{d}_{h}_{seed}_{seed_mh}_Perf_Eps_{0.12}_StopStag_{no_imp_limit}__MHNoRestart_RandDemand_RandStartHeli_TimeDay0700_1900.txt"
    print(perf_over_time)
    os.makedirs("/tmp/results/", exist_ok=True)
    with open(f"/tmp/results/MH_LOGs_{i}_{l}_{d}_{h}_{seed}_{seed_mh}.txt", "w") as text_file:
        text_file.write(logs)

    np.savetxt(expe_name_perf, np.array(perf_over_time), fmt="%s", delimiter=",")
    np.savetxt(expe_name, res, fmt="%s", delimiter=",")

    np.savetxt("/tmp/results/DONE", np.array([]))
    print("Done.")



if __name__ == "__main__":

    main()
