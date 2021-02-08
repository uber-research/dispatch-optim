
import numpy as np
import click
import utils_electric
from routing_charging import optim_electric, evaluation
import time
from prettytable import PrettyTable

@click.command()
@click.argument("v", type=int)
@click.argument("s", type=int)
@click.argument("r", type=int)
def main(v, s, r):
  """
      VNS on benchmark problem instances
      Args:
          v: number of evtol
          s: number of skyports
          r: number of demands
  """
  N = 20 #number of VNS runs for one instance
  timeout = 60 * 30 # 30 minutes timeout
  # get problem instance
  problem_instance = utils_electric.get_problem_instance(s = s, v = v, r = r)
  # Running VNS N times - storing anytime results and final results for comparison
  caches = []
  services = []
  fast_charges = []
  charges = []
  cost_per_service = []
  times_vns = []
  found_valid = []
  print("\nRunning UAM-VNS\n")
  for seed in range(N):
    print(f"=====Run with seed {seed}. ======")
    np.random.seed(seed)
    solution, best_sol, staging_sol, random_sol = utils_electric.init_vns(s,
                                                                          v,
                                                                          r,
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
                                                                    verbose=True,
                                                                    J = 20,
                                                                    incr=100,
                                                                    decr=4,
                                                                    first_best=False)
    print(f"Meta Heuristic VNS done in : {vns_time} seconds. Move per second : {move / vns_time}.")
    # Caching results
    caches.append(anytime_cache)
    times_vns.append(vns_time)
    services.append(round(1 - best_sol.unserved_count / r, 3) * 100)
    cost_per_service.append((best_sol.routing_cost + best_sol.electricity_cost) / (r - best_sol.unserved_count))
    nb_charging = utils_electric.count_all_charges_vns(best_sol, problem_instance)
    print(f"Number of charges : {nb_charging}.")
    charges.append(nb_charging)
    fast_charges.append(round(best_sol.fast_charge_count / nb_charging, 3) * 100)
    found_valid.append(best_sol.violation_tot < 0.3)
  # Mean values are reported and std in parenthesis.
  new_result = f"\n({v}, {s}, {r}), {round(np.mean(services), 3)} ({round(np.std(services), 3)}), {round(np.mean(charges), 3)} ({round(np.std(charges), 3)}) ({round(np.mean(fast_charges), 3)} ({round(np.std(fast_charges), 3)})), {round(np.mean(times_vns), 3)} ({round(np.std(times_vns), 3)}), {round(np.mean(cost_per_service), 3)} ({round(np.std(cost_per_service), 3)}), {np.mean(found_valid)} ({np.std(found_valid)});"
  print(new_result)
  #-- logging to Global gurobi results
  with open("src/logs_results/VNS_output/VNS_results", "a") as file_object:
    file_object.write(new_result)
  print("New line of result added to global file.")
  #-- Now logging Anytime solution to output file
  print("Caching anytime")
  output_anytime = f"src/logs_results/VNS_output/anytime_output_{s}_{v}_{r}"
  output_cps = f"src/logs_results/VNS_output/cps_output_{s}_{v}_{r}"
  all_tables_anytime = "ANYTIME CACHE"
  for cache in caches:
    x = PrettyTable()
    x.field_names = ["Service", "Fast Charges", "Cost per Service", "Time"]
    for i in range(cache.shape[0]):
      row = []
      for c in range(cache.shape[1]):
        val = f"{cache[i, c]}"
        row.append(val)
      x.add_row(row)
    all_tables_anytime += "\n"
    all_tables_anytime += str(x)
  # writing results
  with open(output_anytime, 'w') as out:
    out.write(all_tables_anytime)

  with open(output_cps, 'w') as out:
    out.write(f"{cost_per_service}")
  print("All files written \n Done.")


if __name__ == "__main__":
  main()
