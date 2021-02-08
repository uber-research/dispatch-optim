#####
# Module intended to run locally outside of container.
# Because Gurobi license is used. Therefore it is
# not bazel-compliant on purpose.
#
#####
import sys
sys.path.append("/Users/mehdi/ATCP/test-expe/src/electric/")
import numpy as np
import click
import logging
import utils_electric
import pooling.pooling_utils as pooling_utils
import pooling.optim_pooling as optim_pooling
import routing_charging.optim_electric as optim_electric
import time

@click.command()
@click.argument("n", type=int)
def main(n):
  """ Runs Gurobi on benchmark instance and saves
      results.
      Write summary result in file.
      ----
      Export minizinc to path: export PATH=/Applications/MiniZincIDE.app/Contents/Resources:$PATH

      -----
      Args:


      Returns:

  """
  print("Starting Gurobi Benchmark Expe Pooling")
  timeout = 60 * 30 # 30 minutes timeout
  # get pooling instance
  N_Requests = []
  WT_regular = []
  WT_premium = []
  Loads = []
  Time = []
  for seed in range(20):
    pooling_instance = pooling_utils.get_pooling_instance(n, seed=seed)
    conflicts = pooling_utils.get_conflicts(pooling_instance)
    # get mzn file and output locations
    model_path, output_path = pooling_utils.generate_pooling_mzn(pooling_instance, conflicts, tag=f"_{seed}")
    output_path += f"_seed_{seed}"
    # running model on gurobi
    flat_time = optim_electric.run_gurobi(model_path, timeout, output_path)
    all_sol, time_pool = utils_electric.process_minizinc_output(output_path, flat_time, timeout, pool=True)
    y = all_sol[-1].y
    # -- store metrics for this run
    avg_load, loads = pooling_utils.compute_avg_load(all_sol[-1], pooling_instance)
    wt_premium, wt_regular = pooling_utils.compute_wt(all_sol[-1], pooling_instance)
    WT_regular += wt_regular
    WT_premium += wt_premium
    Loads += loads
    N_Requests.append(np.sum(y))
    Time.append(time_pool - flat_time)

  #-- Write results
  new_line = f"\n{n} & {round(np.mean(N_Requests), 1)} $\pm$ {round(np.std(N_Requests), 1)}"
  new_line += f" & {round(np.mean(Loads), 1)} $\pm$ {round(np.std(Loads), 1)}"
  new_line += f" & {round(np.mean(WT_premium), 1)} $\pm$ {round(np.std(WT_premium), 1)}"
  new_line += f" & {round(np.mean(WT_regular), 1)} $\pm$ {round(np.std(WT_regular), 1)}"
  new_line += f" & {round(np.mean(Time), 1)} $\pm$ {round(np.std(Time), 1)};"

  print(new_line)
  with open("src/logs_results/Gurobi_output_pool/Gurobi_results_pool", "a") as file_object:
    file_object.write(new_line)
  print("New line of result added to global file.")

if __name__ == "__main__":
  main()
