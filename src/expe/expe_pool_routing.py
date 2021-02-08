import sys
sys.path.append('/project/src/')
import joint_optim
import numpy as np
import click
import time
import os


@click.command()
@click.argument("i", type=int)
@click.argument("l", type=int)
@click.argument("h", type=int)
@click.argument("n", type=int)
@click.argument("seed", type=int)
@click.argument("seed_search", type=int)
def main(i, l, h, n, seed, seed_search):
    """ This expe runs the joint optimizer. The goal is to see how the search for better schedule evolves.


        Args:
              j: id of experiments
              l: int, number of skyports to simulate in the infrastructure
              h: int, number of helicopters to simulate for the fleet
              n: int, number of individual demands to simulate
              seed: int, seed for random demands generation and MH
              sedd_search: int, seed for search in TS

        Returns:

            None
    """
    print(f"Starting with : {i, l, h, n, seed, seed_search}")
    print("")
    header = [
        "Obj MH",
        "Valid MH",
        "Served MH"]
    res = np.array(header).reshape((1, len(header)))

    print("Simulating common data...")
    common_data = joint_optim.sim_common_data(l, h, seed=9)
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
    print(logs_nb_poss)

    # ---
    print("Starting to explore space of possible using MH to probe pooling options...")
    best, logs = joint_optim.heuristic_search(common_data, possible_set, legs, 10)

    row = logs

    res = np.concatenate((res, np.array(row).reshape(len(logs), len(header))), axis=0)
    print("Finished current expe. Caching results...")

    expe_name = f"/tmp/results/Expe_Params_JOINT_{i}_{l}_{n}_{h}_{seed}_{seed_search}_Eps_{0.12}_StopStag_MHNoRestart_RandDemand_RandStartHeli_TimeDay0700_1900.txt"
    expe_name_poss = f"/tmp/results/LOGS_N_Poss_JOINT_{i}_{l}_{n}_{h}_{seed}_{seed_search}_Eps_{0.12}_StopStag_MHNoRestart_RandDemand_RandStartHeli_TimeDay0700_1900.txt"

    print(res)
    os.makedirs("/tmp/results/", exist_ok=True)
    np.savetxt(expe_name, res, fmt="%s", delimiter=",")
    np.savetxt(expe_name_poss, logs_nb_poss, fmt="%s", delimiter=",")
    print("Finished.")
    np.savetxt("/tmp/results/DONE", np.array([]))
    print("Done.")




if __name__ == "__main__":
  main()