import sys
sys.path.append('/project/src/')
import numpy as np
import os
import click
from prettytable import PrettyTable


@click.command()
@click.argument("path", type=str)
@click.argument("filename", type=str)
def main(path, filename):
  """
        This function will read every file in path and retrieve results and sum them up in txt file.
        The files that are to be parsed by this script should be txt files in the csv format
        The first row should contains fields names and the second associated values.

        Example of file content :

                        obj mh, time
                        600, 32
    Args :
            path : /results/...
            filename : usually : "Expe_Params_VAR_MVP_", part of filenames before the parameters
    Returns:
            None

  """

  def handle_perf(params, f, thresholds, result_perf):
    """
        Deals with files that contain Perf in name:
        Thos files contains more lines instead of just 2.
        Intended to be used for summing up perf_over_time logs from src/optim.META_HEURISTIC.VNS

        Args:
                params: list of params
                f: str, file name
                thresholds: list, contains int thresholds for parsing table
                result_perf: dict, nested. config, field, treshold.

        Returns:
                updated result_perf


    """
    config = (params[1], params[2], params[3])
    t = open(path + f)
    res = t.read()
    res = res.split("\n")
    tab = [ele.split(",") for ele in res if ele != '']
    if len(tab) == 0:
        start = f.find("_Perf")
        fb = f[:start] + f[start+5:]

        tb = open(path + fb)
        resb = tb.read()

        fieldsb = resb.split("\n")[0].split(",")
        resb = resb.split("\n")[1].split(",")
        tab = []
        for i, c in enumerate(fields):
            if c == "Obj MH":
                tab += [resb[i], '0']
            if c == "Served MH":
                tab += [resb[i]]
        tab = [tab]

    if config not in result_perf:
        result_perf[config] = {ch : {tr : [] for tr in thresholds} for ch in ["Obj", "Served"]}
        for trp in thresholds:

            best_obj = min([float(e[0]) for e in tab if float(e[1]) <= trp], default=np.nan)
            best_served = max([float(e[2]) for e in tab if float(e[1]) <= trp], default=0)
            result_perf[config]["Obj"][trp].append(best_obj)
            result_perf[config]["Served"][trp].append(best_served)
    else:
        for trp in thresholds:

            best_obj = min([float(e[0]) for e in tab if float(e[1]) <= trp], default=np.nan)
            best_served = max([float(e[2]) for e in tab if float(e[1]) <= trp], default=0)
            result_perf[config]["Obj"][trp].append(best_obj)
            result_perf[config]["Served"][trp].append(best_served)
    return result_perf

  files = os.listdir(path)
  has_perf = False #set to true if perf files are to be treated.
  results = {}
  result_perf = {}
  thresholds = [2, 5, 15, 30, 60, np.inf]
  configs = []
  for f in files:
      start = f.find(filename) + len(filename)
      end = f.find("_Eps")
      params = f[start:end].split("_")
      if not (filename) in f:
          continue
      if params == [""]:
          continue
      if 'Perf' in params:
          has_perf = True
          result_perf = handle_perf(params, f, thresholds, result_perf)
          continue
      config = (params[1], params[2], params[3])
      try:
          int(params[1])
      except:
          continue
      configs.append(config)
      t = open(path + f)
      res = t.read()
      fields = res.split("\n")[0].split(",")
      res = res.split("\n")[1].split(",")

      if config not in results:
          results[config] = {c: [] for c in fields}
          for i, c in enumerate(fields):

              results[config][c].append(float(res[i]))
      else:
          for i, c in enumerate(fields):
              if c == "Obj MH" and float(res[i]) < 0:
                  print(f)
                  print(float(res[i]))
              results[config][c].append(float(res[i]))
  x = PrettyTable()
  x.field_names = ["Config"] + fields
  for c in set(configs):
    row = [c]
    for ch in fields:
        val = f"{round(np.mean(results[c][ch]), 2)} ( {round(np.std(results[c][ch]), 2)} ) "
        row.append(val)
    x.add_row(row)

  with open('/results/Performance_logs/Expe_joint_vs_seq_fin.txt', 'w') as out:
      out.write(str(x))
  if has_perf:
    x_perf = PrettyTable()
    x_perf.field_names = ["Config"] + [f"Served : {tr}" for tr in thresholds] + [f"Obj : {tr}" for tr in thresholds]
    for c in set(configs):
        row = [c]
        for tr in thresholds:
            if len(result_perf[c]['Served'][tr]) > 0:
                val = f"{round(np.nanmean(result_perf[c]['Served'][tr]), 4)}"
            else:
                val = 0
            row.append(val)
        for tr in thresholds:
            if len(result_perf[c]['Obj'][tr]) > 0:
                val = f"{round(np.nanmean(result_perf[c]['Obj'][tr]), 4)}"
            else:
                val = 0
            row.append(val)
        x_perf.add_row(row)
        with open('/results/Performance_logs/Expe_res_speed_test_perf.txt', 'w') as out:
            out.write(str(x_perf))

if __name__ == "__main__":
    main()