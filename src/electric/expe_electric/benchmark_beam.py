import sys
sys.path.append("/Users/mehdi/ATCP/test-expe/src/electric/")
import numpy as np
import click
import pooling.pooling_utils as pooling_utils
import pooling.optim_pooling as optim_pooling
import time
import utils_electric

ID = 0
MEAN_ARRIVAL = 1
NPAX = 2
QUANT = 3
MAX_DEP = 4
CLASS = 5

@click.command()
@click.argument("n", type=int)
def main(n):
  """ Runs Beam Search on benchmark instances and saves
      results. POOLING
      Write summary result in file.

      -----
      Args:


      Returns:

  """
  print("Starting Beam Search benchmark for Pooling")
  N_Requests = []
  WT_regular = []
  WT_premium = []
  Loads = []
  Time = []
  for seed in range(20):
    print(f"Starting Beam Search on instance {n} - {seed}")
    pooling_instance = pooling_utils.get_pooling_instance(n, seed=seed)
    beam = pooling_utils.init_beam(pooling_instance, beam_width=1000)
    conflicts = pooling_utils.get_conflicts(pooling_instance)
    time_bs = optim_pooling.beam_search(beam, conflicts)
    stats = pooling_utils.compute_statistics(beam, pooling_instance)
    N_Requests.append(stats.best_n_group)
    WT_premium += stats.waiting_premium
    WT_regular += stats.waiting_regular
    Loads += list(stats.loads[stats.loads>0])
    Time.append(time_bs)

  new_line = f"\n{n} & {round(np.mean(N_Requests), 1)} $\pm$ {round(np.std(N_Requests), 1)}"
  new_line += f" & {round(np.mean(Loads), 1)} $\pm$ {round(np.std(Loads), 1)}"
  new_line += f" & {round(np.mean(WT_premium), 1)} $\pm$ {round(np.std(WT_premium), 1)}"
  new_line += f" & {round(np.mean(WT_regular), 1)} $\pm$ {round(np.std(WT_regular), 1)}"
  new_line += f" & {round(np.mean(Time), 3)} $\pm$ {round(np.std(Time), 3)};"


  #-- Write results
  print(new_line)
  with open("src/logs_results/beam_search_output/beam_search_results", "a") as file_object:
    file_object.write(new_line)
  print("New line of result added to global file.")





if __name__ == "__main__":
  main()
