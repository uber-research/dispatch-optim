import sys
sys.path.append('/project/src/')
import joint_optim
import numpy as np
import click
import time
import os
import utils_expe as ue
import utils_pooling as up
import joint_optim
import optim_incomplete_pooling as incomplete_pooling



@click.command()
@click.argument("j", type=int)
@click.argument("loc", type=int)
@click.argument("h", type=int)
@click.argument("n", type=int)
@click.argument("seed", type=int)
@click.argument("seed_search", type=int)
def main(j, loc, h, n, seed, seed_search):
    """ This function will run an experiment to evaluate our global optimizer against sequential optim.
        1- Running global optimizer with Tree Search for pooling
        2- Running sequentially RS and MH.
        Experiments results will be stored on bucket dispatch-optim/

        Args:
              j: id of experiments
              loc: int, number of skyports to simulate in the infrastructure
              h: int, number of helicopters to simulate for the fleet
              n: int, number of individual demands to simulate
              seed: int, seed for random demands generation and MH
              sedd_search: int, seed for search in TS

        Returns:
              Save results in .txt files under a format which will be acceptable by the src/logs_results/log_parser module.
    """
    print(f"Starting with : {j, loc, h, n, seed, seed_search}")
    print("")

    #running sequentially
    header = [
        "Obj Joint",
        "Time Joint",
        "Served Joint",
        "Obj Seq",
        "Time Seq",
        "Served Seq"]


    res = np.array(header).reshape((1, len(header)))
    row_out = []

    # -----
    ###### Joint optim
    # -----
    print("Simulating common data...")
    common_data = joint_optim.sim_common_data(loc, h, seed=9)
    legs = []
    for s1 in common_data['skyports']:
      for s2 in common_data['skyports']:
        if s1 != s2:
          legs.append((s1, s2))

    table_legs = np.array([e[0] + e[1] for e in legs]).reshape((1, len(legs)))

    # ---
    print("Simulating individuals demands....")
    data_demands = joint_optim.sim_data(common_data["T"], legs, n, seed=seed)

    # ---
    print("Building set of feasible partitions to explore....")
    np.random.seed(seed_search)
    start = time.time()
    possible_set, nb_possible = joint_optim.pool_all(data_demands)
    print(f"Finished building set of possible in {round(time.time() - start, 3)} seconds.")

    possible_legs_nb = []
    for leg in possible_set:
      possible_legs_nb.append(len(possible_set[leg]))
      print(f"Possible pooling for leg {leg} : {len(possible_set[leg])}")
    logs_nb_poss = np.concatenate((table_legs, np.array(possible_legs_nb).reshape(1, len(legs))), axis=0)
    # print("")
    # print(possible_set)
    # --- Caching results
    print("Starting to explore space of possible using MH to probe pooling options...")
    best, logs = joint_optim.heuristic_search(common_data, possible_set, legs, 10)

    row_out += [best[1], round(time.time() - start, 3), best[2]]

    # -----
    ###### Joint optim
    # -----
    deltas = {0: 0.1, 1: 0.2}
    delta = 15
    possible_set_seq = {l : [] for l in legs}
    start = time.time()
    for l in legs:

        points = []
        for i in range(data_demands[l]["tab_points"].shape[0]):
          row = data_demands[l]["tab_points"][i, :]
          points.append((row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
        start = time.time()
        pooling = incomplete_pooling.MATCHING_HEURI(capacity=4,
                                                    delta=delta,
                                                    distri_new_demands=np.array([9, 7, 6]),
                                                    p_type=0.2,
                                                    T=common_data["T"],
                                                    deltas=deltas,
                                                    leg="L")
        nest = pooling.tree(points, pooling.capacity, pooling.delta)
        G, F = pooling.get_groups(nest)
        possible_set_seq[l].append(tuple(F.values()))
    # print("")
    # print(possible_set_seq)
    R_seq = joint_optim.create_demand(possible_set_seq, {l:0 for l in legs}, common_data['fly_time'], verbose=False)
    elements = []
    samples = [s for s in range(6, 15)]
    q = joint_optim.runInParallel(samples, joint_optim.MH_probe, R_seq, common_data)
    for i in range(len(samples)):
        elements.append(q.get())
    best_seq = sorted(elements, key=lambda x: - x[2])[0]
    row_out += [best_seq[1], round(time.time() - start, 3), best_seq[2]]
    # --- Caching results

    print(row_out)

    res = np.concatenate((res, np.array(row_out).reshape(1, len(header))), axis=0)
    print("Finished current expe. Caching results...")

    expe_name = f"/tmp/results/Expe_Params_JOINT_SEQ_{j}_{loc}_{n}_{h}_{seed}_{seed_search}_Eps_{0.12}_StopStag_MHNoRestart_RandDemand_RandStartHeli_TimeDay0700_1900.txt"

    print(res)
    os.makedirs("/tmp/results/", exist_ok=True)
    np.savetxt(expe_name, res, fmt="%s", delimiter=",")
    print("Finished.")
    np.savetxt("/tmp/results/DONE", np.array([]))
    print("Done.")


if __name__ == "__main__":
  main()