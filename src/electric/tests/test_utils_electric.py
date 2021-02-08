"""
    This module contains unit test for the ../utils_electric.py module

"""
import unittest
import utils_electric
import numpy as np
import logging

class TestUtilsElectric(unittest.TestCase):
    """Unittest class intended for the module utils_electric.py

       When input / output checks are not possible/relevant, tests will
       check if the function run smoothly without creating error when called
       and returns the expected type of data.


    """

    def test_infra(self):
        """Test for the simulation of infrastructure"""
        problem_instance = utils_electric.ProblemInstance()
        self.assertIsInstance(problem_instance.costs, np.ndarray)

    def test_time_stamp(self):
        """Test for the format time function"""
        date = utils_electric.format_time(50)
        self.assertIsInstance(date, str)
        self.assertEqual(utils_electric.format_time(0), '07:00')
        self.assertEqual(utils_electric.format_time(30), '07:30')
        self.assertEqual(utils_electric.format_time(720), '19:00')

    def test_time_compatible(self):
        """Test for process_compatibility function"""
        problem_instance = utils_electric.ProblemInstance()
        self.assertIsInstance(problem_instance.costs, np.ndarray)
        self.assertIsInstance(problem_instance.time, np.ndarray)
        self.assertIsInstance(problem_instance.energy, np.ndarray)
        self.assertIsInstance(problem_instance.requests, np.ndarray)
        self.assertIsInstance(problem_instance.aircraft, np.ndarray)
        self.assertIsInstance(problem_instance.skyports, np.ndarray)

    def test_default_instance_gen(self):
        """ Test for the simulate_default_problem_instance() function """
        problem_instance = utils_electric.ProblemInstance()
        self.assertIsInstance(problem_instance.costs, np.ndarray)
        self.assertIsInstance(problem_instance.time, np.ndarray)
        self.assertIsInstance(problem_instance.energy, np.ndarray)
        self.assertIsInstance(problem_instance.requests, np.ndarray)
        self.assertIsInstance(problem_instance.aircraft, np.ndarray)
        self.assertIsInstance(problem_instance.skyports, np.ndarray)
        self.assertIsInstance(problem_instance.T, list)
        self.assertIsInstance(problem_instance.min_soc, float)
        self.assertIsInstance(problem_instance.lbda_f, float)
        self.assertIsInstance(problem_instance.lbda_u, float)
        self.assertIsInstance(problem_instance.pe, float)
        self.assertIsInstance(problem_instance.gamma_f, float)
        self.assertIsInstance(problem_instance.gamma_s, float)
        self.assertIsInstance(problem_instance.delta, int)
        self.assertTrue(problem_instance.delta >= 0.)
        self.assertTrue(problem_instance.gamma_s >= 0.)
        self.assertTrue(problem_instance.gamma_f >= 0.)
        self.assertTrue(problem_instance.pe >= 0.)
        self.assertTrue(problem_instance.min_soc >= 0.)
        self.assertTrue(problem_instance.min_soc < 100.)
        self.assertTrue(problem_instance.lbda_f >= 0.)
        self.assertTrue(problem_instance.lbda_u >= 0.)


if __name__ == "__main__":
    unittest.main()