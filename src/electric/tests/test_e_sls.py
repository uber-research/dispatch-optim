"""
    This module contains unit tests for the ../optim_electric.py module
    Specifically for the ElectricSls class in it.
    The goal here have elementary tests for the ElectricSls class.

"""

import unittest
import utils_electric
import optim_electric
import numpy as np
import logging

class TestElectricSls(unittest.TestCase):
    """Unittest class intended for the module optim_electric.py
       Specifically for the ElectricSls class in it.
       When input / output checks are not possible/relevant, tests will
       check if the function run smoothly without creating error when called
       and returns the expected type of data.


    """

    def test_init(self):
        """Test if ElectricSls instantiate without errors.
           And also test for instantiated attributes.
        """
        problem_instance = utils_electric.ProblemInstance()
        sls = optim_electric.ElectricSls(problem_instance.costs,
                                         problem_instance.time,
                                         problem_instance.energy,
                                         problem_instance.requests,
                                         problem_instance.aircraft,
                                         problem_instance.skyports,
                                         problem_instance.time_compatible,
                                         problem_instance.gamma_s,
                                         problem_instance.gamma_f,
                                         problem_instance.pe,
                                         problem_instance.delta,
                                         problem_instance.lbda_f,
                                         problem_instance.min_soc,
                                         problem_instance.T)
        self.assertIsInstance(sls.empty_sol, np.ndarray)
        self.assertTrue(np.sum(sls.empty_sol) == 0)
        self.assertTrue(np.sum(sls.H) == problem_instance.aircraft.shape[0])
        self.assertTrue(np.sum(sls.len_encod) ==  2 * (problem_instance.requests.shape[0] - problem_instance.aircraft.shape[0]))

    def test_init_heuristic(self):
        """Test the init heuristic provide an encoding of the right format
        """
        problem_instance = utils_electric.ProblemInstance()
        sls = optim_electric.ElectricSls(problem_instance.costs,
                                         problem_instance.time,
                                         problem_instance.energy,
                                         problem_instance.requests,
                                         problem_instance.aircraft,
                                         problem_instance.skyports,
                                         problem_instance.time_compatible,
                                         problem_instance.gamma_s,
                                         problem_instance.gamma_f,
                                         problem_instance.pe,
                                         problem_instance.delta,
                                         problem_instance.lbda_f,
                                         problem_instance.min_soc,
                                         problem_instance.T)

        np.random.seed(6)
        init_heuri = sls.init_heuristic_random()
        self.assertIsInstance(init_heuri, np.ndarray)
        self.assertTrue(len(init_heuri) == (problem_instance.requests.shape[0] - problem_instance.aircraft.shape[0]) * 2 * problem_instance.aircraft.shape[0])


    def test_solution_feasibility(self):
        """Test the feasibility check behaves correctly :
              - an empty solution must be valid
              - solutions generated by initial heuristic must be valid
              - All ones solutions must not be valid
        """
        problem_instance = utils_electric.ProblemInstance()
        sls = optim_electric.ElectricSls(problem_instance.costs,
                                         problem_instance.time,
                                         problem_instance.energy,
                                         problem_instance.requests,
                                         problem_instance.aircraft,
                                         problem_instance.skyports,
                                         problem_instance.time_compatible,
                                         problem_instance.gamma_s,
                                         problem_instance.gamma_f,
                                         problem_instance.pe,
                                         problem_instance.delta,
                                         problem_instance.lbda_f,
                                         problem_instance.min_soc,
                                         problem_instance.T)
        # since init heuristic is random by design
        # run it a few times with different seeds.
        for s in range(6):
          np.random.seed(s)
          init_heuri = sls.init_heuristic_random()
          self.assertTrue(sls.feasible_fast(init_heuri, sls.len_encod, sls.H, problem_instance.time_compatible, sls.mask))
        all_ones = np.ones(init_heuri.shape)
        self.assertFalse(sls.feasible_fast(all_ones, sls.len_encod, sls.H, problem_instance.time_compatible, sls.mask))


    def test_cost_computation_aircraft(self):
        """Test for the function that computes the cost for one aircraft in a solution.
           Check for :
                - The output format and type
                - Violations should be positive (costs may be negative)
                - Empty sol should have 0 cost

        """
        problem_instance = utils_electric.ProblemInstance()
        sls = optim_electric.ElectricSls(problem_instance.costs,
                                         problem_instance.time,
                                         problem_instance.energy,
                                         problem_instance.requests,
                                         problem_instance.aircraft,
                                         problem_instance.skyports,
                                         problem_instance.time_compatible,
                                         problem_instance.gamma_s,
                                         problem_instance.gamma_f,
                                         problem_instance.pe,
                                         problem_instance.delta,
                                         problem_instance.lbda_f,
                                         problem_instance.min_soc,
                                         problem_instance.T)

        init_heuri = sls.init_heuristic_random()
        c = sls.trace_cost_heli(init_heuri, 0)
        c_empty = sls.trace_cost_heli(sls.empty_sol, 0)
        self.assertIsInstance(c, list)
        self.assertTrue(c[1] >= 0)
        self.assertTrue(c_empty[0] == c_empty[1] == 0)

    def test_cost_computation_sol(self):
        """Test for the function that computes the cost of an entire solution encoding."""
        problem_instance = utils_electric.ProblemInstance()
        sls = optim_electric.ElectricSls(problem_instance.costs,
                                         problem_instance.time,
                                         problem_instance.energy,
                                         problem_instance.requests,
                                         problem_instance.aircraft,
                                         problem_instance.skyports,
                                         problem_instance.time_compatible,
                                         problem_instance.gamma_s,
                                         problem_instance.gamma_f,
                                         problem_instance.pe,
                                         problem_instance.delta,
                                         problem_instance.lbda_f,
                                         problem_instance.min_soc,
                                         problem_instance.T)

        init_heuri = sls.init_heuristic_random()
        c = sls.compute_cost(init_heuri)
        c_inf = sls.compute_cost(np.ones(init_heuri.shape))
        self.assertTrue(c[2] >= 0)
        self.assertIsInstance(c[1], dict)
        self.assertTrue(c_inf[0] == np.inf)

    def test_neighbourhoods(self):
        """
            Testing neighbourhoods function run correctly
            and return valid solutions.
        """
        problem_instance = utils_electric.ProblemInstance()
        sls = optim_electric.ElectricSls(problem_instance.costs,
                                         problem_instance.time,
                                         problem_instance.energy,
                                         problem_instance.requests,
                                         problem_instance.aircraft,
                                         problem_instance.skyports,
                                         problem_instance.time_compatible,
                                         problem_instance.gamma_s,
                                         problem_instance.gamma_f,
                                         problem_instance.pe,
                                         problem_instance.delta,
                                         problem_instance.lbda_f,
                                         problem_instance.min_soc,
                                         problem_instance.T)
        init_heuri = sls.init_heuristic_random()
        unserved, served = sls.update_served_status_fast(init_heuri, sls.n_requests, sls.len_encod)
        # remove req and add fast charge neighbourhood
        rm_rf = sls.rm_rf_neigh(init_heuri, served, unserved)
        self.assertIsInstance(rm_rf[0], np.ndarray)
        self.assertIsInstance(rm_rf[1], float)
        self.assertTrue(sls.feasible_fast(rm_rf[0], sls.len_encod, sls.H, sls.time_compatible, sls.mask))
        # fast charge neighbourhood
        rf = sls.refuel_neigh(init_heuri, served, unserved)
        self.assertIsInstance(rf[0], np.ndarray)
        self.assertIsInstance(rf[1], float)
        self.assertTrue(sls.feasible_fast(rf[0], sls.len_encod, sls.H, sls.time_compatible, sls.mask))
        # swap  2 req neighbourhood
        sw = sls.swap_neigh(init_heuri, served, unserved)
        self.assertIsInstance(sw[0], np.ndarray)
        self.assertIsInstance(sw[1], float)
        self.assertTrue(sls.feasible_fast(sw[0], sls.len_encod, sls.H, sls.time_compatible, sls.mask))
        # shift neighbourhood
        sh = sls.shift_neigh(init_heuri, served, unserved)
        self.assertIsInstance(sh[0], np.ndarray)
        self.assertIsInstance(sh[1], float)
        self.assertTrue(sls.feasible_fast(sh[0], sls.len_encod, sls.H, sls.time_compatible, sls.mask))
        # swap paths neighbourhood
        sp = sls.swap_paths_neigh(init_heuri, served, unserved)
        self.assertIsInstance(sp[0], np.ndarray)
        self.assertIsInstance(sp[1], float)
        self.assertTrue(sls.feasible_fast(sp[0], sls.len_encod, sls.H, sls.time_compatible, sls.mask))





if __name__ == "__main__":
    unittest.main()