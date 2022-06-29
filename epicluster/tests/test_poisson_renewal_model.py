"""Test the code in the module poisson_renewal_model.py.
"""

import math
import unittest
from unittest.mock import patch
import epicluster as ec


class TestPoissonRenewalModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Make simple data for testing
        cls.cases = [1, 2, 3, 4, 5, 6]
        cls.serial_interval = [0.1, 0.9]
        cls.imported_cases = [1, 0, 2, 1, 0, 0]

    def test_init(self):
        model = ec.PoissonModel(self.cases,
                                self.serial_interval,
                                imported_cases=self.imported_cases)

        self.assertEqual(model.all_cases, self.cases)
        self.assertEqual(model.cases, [3, 4, 5, 6])
        self.assertEqual(model.assignments, [0, 0, 0, 0])

        # Check setting sigma
        model = ec.PoissonModel(self.cases,
                                self.serial_interval,
                                imported_cases=self.imported_cases,
                                prior_expected_clusters=1.5)
        self.assertAlmostEqual(
            model.hyper_sigma,
            ec.RestrictedPYEPPF().find_prior_hyperparam(4, num_blocks=1.5))

    def test_marginal_likelihood(self):
        model = ec.PoissonModel(self.cases,
                                self.serial_interval,
                                imported_cases=self.imported_cases)
        mll = model.marginal_likelihood()
        self.assertTrue(math.isfinite(mll))

    def test_update_change_params(self):
        model = ec.PoissonModel(self.cases,
                                self.serial_interval,
                                imported_cases=self.imported_cases)

        model.assignments = [0, 0, 0, 0]
        model.update_change_params()

        # Check that there is one R value
        self.assertEqual(len(model.change_params), 1)
        self.assertTrue(model.change_params[0] > 0)

        model.assignments = [0, 1, 2, 3]
        model.update_change_params()

        # Check that there are four R values
        self.assertEqual(len(model.change_params), 4)
        self.assertTrue(model.change_params[0] > 0)
        self.assertTrue(model.change_params[1] > 0)
        self.assertTrue(model.change_params[2] > 0)
        self.assertTrue(model.change_params[3] > 0)


if __name__ == '__main__':
    unittest.main()
