#####
# Module intended to run locally outside of container.
# Because Gurobi license is used. Therefore it is
# not bazel-compliant on purpose.
#
#####
import sys
sys.path.append("/Users/mehdi/ATCP/test-expe/src/")
import numpy as np
import click

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
  print("Retrieving gurobi CPS")
  results_gurobi = f"src/logs_results/Gurobi_output/Gurobi_results"
  cps_vns = f"src/logs_results/VNS_output/cps_output_{s}_{v}_{r}"
  file_gurobi = open(results_gurobi, "r")
  table_gurobi = file_gurobi.read()
  table_gurobi = table_gurobi.split(";")
  for row in table_gurobi:
    elements = row.split(",")
    config = elements[:3]
    cps = elements[-1]
    if config == [f"\n({v}", f" {s}", f" {r})"]:
      cps_gurobi = float(cps)
      break
  #-- get cps vns
  file_vns = open(cps_vns, "r")
  row_vns = file_vns.read().split(",")
  values = np.array([float(val.strip(" ").strip("[").strip("]")) for val in row_vns])
  cps_gaps = ((values - cps_gurobi) / cps_gurobi) * 100
  print(f"Mean CPS Gap {round(np.mean(cps_gaps), 2)}, std {round(np.std(cps_gaps), 2)}")
  with open("src/logs_results/VNS_output/CPS_Gaps", "a") as file_object:
    file_object.write(f"\n{v, s, r}, {round(np.mean(cps_gaps), 2)} ({round(np.std(cps_gaps), 2)});")
  print("New line of result added to global file.")




if __name__ == "__main__":
  main()
