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
import routing_charging.optim_electric as optim_electric
import time

@click.command()
@click.argument("v", type=int)
@click.argument("s", type=int)
@click.argument("r", type=int)
def main(v, s, r):
  """ Runs Gurobi on benchmark instance and saves
      results.
      Write summary result in file.
      ----
      Export minizinc to path: export PATH=/Applications/MiniZincIDE.app/Contents/Resources:$PATH

      -----
      Args:


      Returns:

  """
  print("Starting Gurobi....")
  timeout = 60 * 30 # 30 minutes timeout
  # get problem instance
  problem_instance = utils_electric.get_problem_instance(s = s, v = v, r = r)
  # get mzn model
  model_path, output_path = utils_electric.generate_mzn(problem_instance, s, v, r)
  flat_time = optim_electric.run_gurobi(model_path, timeout, output_path)
  # Retrieve gurobi metrics
  all_sol, final_time = utils_electric.process_minizinc_output(output_path, flat_time, timeout)
  last_sol = all_sol[-1]
  service = round(1 - last_sol.unserved / r, 3) * 100
  nb_charge = last_sol.get_nb_charge()
  print(f"Nb charge: {nb_charge}.")
  fast_count = round(last_sol.fast_charges / nb_charge, 3) * 100
  time = final_time - flat_time
  routing_cost, elec_cost = last_sol.get_op_cost(problem_instance)
  cost_per_service = (routing_cost + elec_cost) / (r - last_sol.unserved)
  new_result = f"\n({v}, {s}, {r}), {service}, {nb_charge} ({fast_count}), {time}, {cost_per_service};"
  print(new_result)
  #-- logging to Global gurobi results
  with open("src/logs_results/Gurobi_output/Gurobi_results", "a") as file_object:
    file_object.write(new_result)
  print("New line of result added to global file.")


if __name__ == "__main__":
  main()
