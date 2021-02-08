import sys
sys.path.append("/Users/mehdi/ATCP/test-expe/src/electric/")
from figures_paper import illustrations

def main():
  """
      Calling every illusrtations function to generate all figures.

  """
  illustrations.get_arrival_time_figure()
  print("Done.")


if __name__ == "__main__":
  main()
