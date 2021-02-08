from numba import njit
import numpy as np


@njit(inline="always")
def copy_solution(sol_a: object, sol_b: object):
  """ Copy sol_a attributes into sol_b attributes.

      Args:
          sol_a: object
          sol_b: object

      Returns:
          None
  """
  sol_b.assignement[:] = sol_a.assignement[:]
  sol_b.assignementMap[:,:,:] = sol_a.assignementMap[:,:,:]
  sol_b.charging_times[:,:,:] = sol_a.charging_times[:,:,:]
  sol_b.energy_levels[:,:,:] = sol_a.energy_levels[:,:,:]
  sol_b.energy_bought[:,:,:] = sol_a.energy_bought[:,:,:]
  sol_b.violation[:,:,:] = sol_a.violation[:,:,:]
  sol_b.violation_tot = sol_a.violation_tot
  sol_b.unserved_count = sol_a.unserved_count
  sol_b.fast_charge_count = sol_a.fast_charge_count
  sol_b.cost = sol_a.cost
  sol_b.routing_cost = sol_a.routing_cost
  sol_b.electricity_cost = sol_a.electricity_cost
  return None



@njit(inline="always")
def update_penalty(penalty: float,
                   rolling_violation: np.ndarray,
                   incr: float,
                   decr: float,
                   s: float = 1.):
  """ Updates penalty for battery soc violation.
      Additive increase, multiplicative decrease.

      Args:
          penalty: incoming penalty value
          rolling_violation: it is the sum of the violation of the current solution
                            over the last H iterations. H is the size of the rolling
                            window.
          incr: increment value
          decr: decrement value
      Returns:
          new value for penalty, float
  """
  if np.sum(rolling_violation) > s:
    return penalty + incr
  else:
    return penalty / decr


