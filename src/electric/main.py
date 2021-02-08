import numpy as np
import click
import pooling.pooling_utils as pooling_utils

@click.command()
@click.argument("n", type=int)
def main(n):
  """
      Test pooling
  """
  pooling_instance = pooling_utils.get_pooling_instance(n)
  print(pooling_instance.demands)
  pooling_utils.generate_pooling_mzn(pooling_instance)

if __name__ == "__main__":
  main()
