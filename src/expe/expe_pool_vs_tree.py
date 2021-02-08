import sys
sys.path.append('/project/src/')
import joint_optim
import numpy as np
import click
import time
import os
import optim_incomplete_pooling as incomplete_pooling


@click.command()
@click.argument("i", type=int)
@click.argument("l", type=int)
@click.argument("n", type=int)
@click.argument("seed", type=int)
@click.argument("seed_search", type=int)
def main(i, l, n, seed, seed_search):
    """ This expe will compare RS and TS in term of running time and pooling rate.


        Args:
              j: id of experiments
              l: int, number of skyports to simulate in the infrastructure

              n: int, number of individual demands to simulate
              seed: int, seed for random demands generation and MH
              sedd_search: int, seed for search in TS

        Returns:
              None
    """
    h = 2
    l = 2
    print(f"Starting with ranges {i, l, n, seed, seed_search}")
    # results = {n: {"TS time": [], "RS time": [], "TS PR": [], "RS PR": []} for n in [10, 15, 20, 35, 30]}
    header = ["TS time",
              "TS PR",
              "RS time",
              "RS PR",
              "Valid"]
    res = np.array(header).reshape((1, len(header)))
    row_out = []
    common_data = joint_optim.sim_common_data(l, h, seed=seed)
    legs = [(common_data['skyports'][0], common_data['skyports'][1])]

    # print("Simulating individuals demands....")
    data_demands = joint_optim.sim_data(common_data["T"], legs, n, seed=seed)

    # ---
    print("Building set of feasible partitions to explore....")
    np.random.seed(seed_search)
    start = time.time()
    possible_set, nb_possible = joint_optim.pool_all(data_demands)
    #results[n]["TS time"].append(round(time.time() - start, 3))
    row_out.append(round(time.time() - start, 3))
    #print((len(possible_set[legs[0]][0]) / n))
    pr_ts = round(1 - (len(possible_set[legs[0]][0]) / n), 3)
    #print(possible_set[legs[0]][0])
    row_out.append(pr_ts)

    deltas = {0: 0.1, 1: 0.2}
    delta = 15
    points = []
    for i in range(data_demands[legs[0]]["tab_points"].shape[0]):
      row = data_demands[legs[0]]["tab_points"][i, :]
      points.append((row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
    start = time.time()
    pooling = incomplete_pooling.MATCHING_HEURI(capacity=4,
                                                delta=delta,
                                                distri_new_demands=np.array([9,7,6]),
                                                p_type=0.2,
                                                T=common_data["T"],
                                                deltas=deltas,
                                                leg="L")
    nest = pooling.tree(points, pooling.capacity, pooling.delta)
    G, F = pooling.get_groups(nest)
    #results[n]["RS time"].append(round(time.time() - start, 3))
    row_out.append(round(time.time() - start, 3))
    row_out.append(round(1 - (len(F) / n), 3))
    flights_rs = list(F.values())
    flights_rs.sort()
    #print(flights_rs)
    # print(flights_rs)
    rs_g = [[int(e[-2]) for e in li] for li in list(G.values())]
    def check_joint(cont):
      for g in cont:

          if len(g) > 4:
              print(len(g))
              return False

          feas_spread = joint_optim.check_spread(g, data_demands[legs[0]]["tab_points"], -1, 2, 15, joint_optim.aversion, 10)
          if not (feas_spread):
              print("spread")
              print(g)
              return False
          feas_cap = joint_optim.check_cap(g, 4, data_demands[legs[0]]["tab_points"], 4)
          if not (feas_cap):
              print("f cap")
              return False
      return True
    #print(check_joint(rs_g))

    row_out.append(check_joint(rs_g))
    res = np.concatenate((res, np.array(row_out).reshape(1, len(header))), axis=0)

    print("Finished current expe. Caching results...")
    #expe_name = f"/tmp/results/Expe_Params_VAR_{i}_{l}_{d}_{h}_{seed}_{seed_mh}_Eps_{0.12}_StopStag_{no_imp_limit}__MHNoRestart_MaxTimeMilp_{max_runtime}__RandDemand_RandStartHeli_TimeDay0700_1900.txt"
    expe_name = f"/tmp/results/Expe_Params_RS_vs_TS_{i}_{n}_{seed}_{seed_search}_Eps_{0.12}_StopStag_MHNoRestart_RandDemand_RandStartHeli_TimeDay0700_1900.txt"

    print(res)
    os.makedirs("/tmp/results/", exist_ok=True)
    np.savetxt(expe_name, res, fmt="%s", delimiter=",")
    print("Finished.")
    np.savetxt("/tmp/results/DONE", np.array([]))
    print("Done.")



if __name__ == "__main__":
  main()
