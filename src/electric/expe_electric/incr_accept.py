import sys
sys.path.append("/Users/mehdi/ATCP/test-expe/src/electric/")
import numpy as np
import click
import pooling.pooling_utils as pooling_utils
import pooling.optim_pooling as optim_pooling
import time
import utils_electric
from routing_charging import optim_electric, evaluation, search_utils
from online import incremental_pooling, incremental_utils
import matplotlib.pyplot as plt
import seaborn as sns

ORIGIN = 1
DEST = 2
ORIGIN_TIME = 3
DEST_TIME = 4

@click.command()
@click.argument("v", type=int)
@click.argument("s", type=int)
@click.argument("d", type=int)
def main(v, s, d):
  """ Incremental acceptance for VNS.

      -----
      Args:


      Returns:

  """
  print("Starting incremental test")
  seed = 0
  if d > 200:
    seed = 1
  problem_instance = utils_electric.get_problem_instance(s = s, v = v, r = 3)
  legs = [(i, j) for i in range(s) for j in range(s) if i != j]
  leg_mode, instances_pooling = incremental_utils.simulate_all_demands(d, legs, problem_instance)
  print("Request simulated via pooling on all legs")
  init_r = problem_instance.r
  print(f"Initial number of requests : {init_r}")
  np.random.seed(seed)
  solution, best_sol, staging_sol, random_sol = utils_electric.init_vns(s,
                                                                          v,
                                                                          problem_instance.r,
                                                                          problem_instance,
                                                                          evaluation.commit_greedy_cost,
                                                                          evaluation.commit_propagation_new_energy,
                                                                          optim_electric.SolutionSls)


  pen_energy, vns_time, move, anytime_cache = optim_electric.GVNS(solution,
                                                                    best_sol,
                                                                    staging_sol,
                                                                    random_sol,
                                                                    200000, #big enough to ensure timeout or convergence
                                                                    30 * 60, # 30 minutes timeout
                                                                    problem_instance,
                                                                    seed=seed,
                                                                    verbose=False,
                                                                    J = 20,
                                                                    incr=100,
                                                                    decr=4,
                                                                    first_best = False)


  print(f"Meta Heuristic VNS first run done. Time {vns_time}.")
  print(f"Cost : {best_sol.cost}. Service: {round(1 - best_sol.unserved_count/problem_instance.r,2)}")
  cache_acceptance = np.zeros((1, 7))
  cache_pb_instance = incremental_utils.save_pb_instance(problem_instance, "pb_instance")
  for i, ele in enumerate(instances_pooling):
    incremental_utils.save_pooling_ins(ele, f"pooling_{i}")
  cache_sol_init = incremental_utils.save_solution(best_sol, "sol_init")
  additionnal_req = []
  for seed_d in range(20):
    print(f"Looping -- {seed_d}")
    #---- Reloading solutions
    incremental_utils.load_solution(best_sol, "sol_init", cache_sol_init)
    incremental_utils.load_pb_instance(problem_instance, "pb_instance", cache_pb_instance)
    for i, ele in enumerate(instances_pooling):
      incremental_utils.load_pooling_ins(ele, f"pooling_{i}")

    #----
    # print(f"Number of requests : {problem_instance.r}. Cost : {best_sol.cost}. ")
    start_incr = time.time()
    new_requests, old_req_leg, incr_leg = incremental_utils.get_new_requests(d,
                                                                    legs,
                                                                    instances_pooling,
                                                                    leg_mode,
                                                                    problem_instance,
                                                                    seed=seed_d)
    old_best_sol, new_best_sol, staging_sol, random_sol, sol_backup = incremental_utils.remove_and_add(problem_instance,
                                                                              new_requests,
                                                                              incr_leg,
                                                                              best_sol)
    # print(f"Cost : {sol_backup.cost}. Service: {round(1 - sol_backup.unserved_count/problem_instance.r,2)}")
    pooling_time = time.time() - start_incr
    additionnal_req.append(problem_instance.r - init_r)
    # print(f"New requests {problem_instance.r}. Init was {init_r}")
    # print(f"Pooling time is {pooling_time}")
    acceptance = np.zeros((1, 7))
    for seed_vns in range(10):
      search_utils.copy_solution(sol_backup, best_sol)
      search_utils.copy_solution(sol_backup, new_best_sol)
      search_utils.copy_solution(sol_backup, staging_sol)
      search_utils.copy_solution(sol_backup, random_sol)
      pen_energy, vns_time, move, anytime_cache = optim_electric.GVNS(best_sol,
                                                                        new_best_sol,
                                                                        staging_sol,
                                                                        random_sol,
                                                                        200000, #big enough to ensure timeout or convergence
                                                                        30 * 60, # 30 minutes timeout
                                                                        problem_instance,
                                                                        seed=seed_vns,
                                                                        verbose=False,
                                                                        J = 20,
                                                                        incr=100,
                                                                        decr=4,
                                                                        first_best=False,
                                                                        SAT_stop=True)

      accept_status = incremental_utils.process_anytime(anytime_cache, pooling_time)
      # print(f"Best sol with cost {new_best_sol.cost} and unserved : {new_best_sol.unserved_count}, violation {new_best_sol.violation_tot}. || Acceptance : {accept_status}")
      acceptance = np.concatenate((acceptance, accept_status), axis=0)

    id_max = np.argmax(np.max(acceptance, axis=1))
    cache_acceptance = np.concatenate((cache_acceptance, acceptance[id_max,:].reshape(1, 7)), axis=0)

  #-- Caching results
  cache_acceptance = cache_acceptance[1:, :]
  print(cache_acceptance*100)
  cache_acceptance *= 100

  mean_acceptance = np.mean(cache_acceptance, axis=0)
  std_acceptance = np.std(cache_acceptance, axis=0)
  print(mean_acceptance)
  print(std_acceptance)

  #-- Write results
  new_line = f"\n{d}, {init_r},"
  new_line += f"{np.mean(additionnal_req)} ({np.std(additionnal_req)}),"
  for i in range(len(mean_acceptance)):
    new_line += f"{round(mean_acceptance[i], 2)} ({round(std_acceptance[i], 2)}),"
  new_line += ";"
  print(new_line)
  with open("src/logs_results/online_analysis/incremental_accept", "a") as file_object:
    file_object.write(new_line)
  print("New line of result added to global file.")

if __name__ == "__main__":
  main()


