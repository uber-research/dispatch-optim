###
# Illustrations for the Passenger centric UAM paper.
#
###
import sys
sys.path.append("/Users/mehdi/ATCP/test-expe/src/electric/")
import matplotlib.pyplot as plt
import numpy as np

def get_arrival_time_figure():
  """ """
  def lognorm(mu=0, sig=1):
    x = np.linspace(0.1, 50, 1000)
    return x, 1./(sig*x*np.sqrt(2*np.pi)) * (np.exp(-np.log(x)**2/(2*sig**2)))

  x, y = lognorm(0, .65)
  # plt.figure(figsize=(15, 8))
  fig, ax = plt.subplots(figsize=(20,9))
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  alpha_bt = 0
  plt.plot(x, y, color='blue', alpha=0.8)
  sample = np.random.lognormal(0, 0.65, 1000)
  quant = np.quantile(sample, 0.95)
  mask = x > quant
  plt.fill_between(x[mask], 0, y[mask], alpha=alpha_bt)
  plt.axvspan(xmin=x[mask][0], xmax=x[mask][0] + 11, ymin=0.0, ymax=0.025, facecolor='blue', alpha=0.5)

  plt.plot(x+5, y, color='blue', alpha=0.8)
  xp = x+5
  plt.fill_between(xp[mask], 0, y[mask], color='blue', alpha=alpha_bt)
  plt.axvspan(xmin=xp[mask][0], xmax=xp[mask][0] + 11, ymin=0.0, ymax=0.025, facecolor='blue', alpha=0.5, label="Regular")

  plt.plot(x+15, y, color='orange', alpha=0.8)
  xp = x+15
  plt.fill_between(xp[mask], 0, y[mask], color='orange', alpha=alpha_bt)

  plt.axvspan(xmin=xp[mask][0], xmax=xp[mask][0]+6, ymin=0.025, ymax=0.05, facecolor='orange', alpha=0.5)

  plt.plot(x+3, y, color='orange', alpha=0.8)
  xp = x+3
  plt.fill_between(xp[mask], 0, y[mask], color='orange', alpha=alpha_bt)

  plt.xticks(ticks=[0, 5, 10, 15, 20, 25], labels=["8:00", "8:10", "8:20", "8:30", "8:40", ""])
  ax.tick_params(axis='both', labelsize=18, pad=16)
  plt.axvspan(xmin=xp[mask][0], xmax=xp[mask][0]+6, ymin=0.025, ymax=0.05, facecolor='orange', alpha=0.5, label="Premium")

  plt.legend(prop={'size': 20}, loc='upper right')
  plt.xlim(-2, 25)
  plt.ylim(-0.0, 0.8)
  # plt.xlabel("Arrival times", fontdict=dict(size=19))
  plt.ylabel("Likelihood", fontdict=dict(size=21))
  plot_loc = f"src/logs_results/analysis/Constraints_arrival_times"
  plt.savefig(plot_loc)