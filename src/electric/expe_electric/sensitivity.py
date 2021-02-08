import sys
sys.path.append("/Users/mehdi/ATCP/test-expe/src/electric/")
import numpy as np
import click
import pooling.pooling_utils as pooling_utils
import pooling.optim_pooling as optim_pooling
import time
import utils_electric
import matplotlib.pyplot as plt
import seaborn as sns

ID = 0
MEAN_ARRIVAL = 1
NPAX = 2
QUANT = 3
MAX_DEP = 4
CLASS = 5

@click.command()
@click.argument("n", type=int)
def main(n):
  """ Runs Beam Search on one benchmark instance and saves
      results and study sensitivity.

      -----
      Args:


      Returns:

  """
  ar_grid = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
  mwp_grid = [t for t in range(15, 26)]

  #--- Alpha impact Depracated
  # for prop_premium in [0.2, 0.5, 0.8]:
  #   mwr = 25
  #   mwp = 25
  #   pooling_utils.impact_alpha(ar_grid,
  #                             n,
  #                             optim_pooling.beam_search,
  #                             mwr = mwr,
  #                             mwp = mwp,
  #                             prop_premium=prop_premium)
  #-- MW change Deprecated
  # for mwp in [15, 20]:
  #   mwr = 25
  #   prop_premium = 0.5
  #   pooling_utils.impact_alpha(ar_grid,
  #                             n,
  #                             optim_pooling.beam_search,
  #                             mwr = mwr,
  #                             mwp = mwp,
  #                             prop_premium=prop_premium)

  #-- Mixity impact
  for prop_premium in [0.2, 0.5, 0.8]:
    mwr = 25
    mwp = 15
    pooling_utils.impact_mixity(ar_grid,
                                n,
                                optim_pooling.beam_search,
                                mwr = mwr,
                                mwp = mwp,
                                prop_premium=prop_premium)


  for mwr in [25]:#mvr in [25, 35, 45]:
    prop_premium = 0.5
    pooling_utils.impact_hard_constraints(mwp_grid,
                                          ar_grid,
                                          n,
                                          optim_pooling.beam_search,
                                          mwr=mwr,
                                          prop_premium=prop_premium)

if __name__ == "__main__":
  main()
