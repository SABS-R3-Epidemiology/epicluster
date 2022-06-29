"""Test the code in the module posterior.py.
"""

import math
import unittest
from unittest.mock import patch
import epicluster as ec


class TestPosterior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Make simple data for testing
        cls.cases = [1, 2, 3, 4, 5, 6]
        cls.serial_interval = [0.1, 0.9]
        cls.imported_cases = [1, 0, 2, 1, 0, 0]
        model = ec.PoissonModel(cls.cases,
                                cls.serial_interval,
                                imported_cases=cls.imported_cases)
        cls.model = model

    def test_init(self):
        sampler = ec.MCMCSampler(self.model, 2)
        self.assertTrue(len(sampler.models), 2)

        sampler = ec.MCMCSampler(self.model, 10)
        self.assertTrue(len(sampler.models), 10)

        # check that the models are separate objects
        sampler = ec.MCMCSampler(self.model, 10)
        sampler.models[0].assignments = [0, 1, 2, 0]
        # If they were copied correctly, the second model should still have
        # the original assignments
        self.assertEqual(sampler.models[1].assignments, [0, 0, 0, 0])

    def test_run_mcmc(self):
        sampler = ec.MCMCSampler(self.model, 2)

        # Check that it fails if no stopping criteria is provided
        with self.assertRaises(ValueError):
            params_chain, assign_chain, clusters_chain = sampler.run_mcmc()

        # Check with a fixed number of mcmc samplers
        params_chain, assign_chain, clusters_chain = \
            sampler.run_mcmc(num_mcmc_samples=5)

        self.assertEqual(len(params_chain), 5)
        self.assertEqual(len(assign_chain), 5)
        self.assertEqual(len(clusters_chain), 5)

        # Check with an Rhat threshold (though it will hit the max_mcmc here)
        params_chain, assign_chain, clusters_chain = \
            sampler.run_mcmc(Rhat_thresh=1.01,
                             max_mcmc=5)

        self.assertEqual(len(params_chain), 5)
        self.assertEqual(len(assign_chain), 5)
        self.assertEqual(len(clusters_chain), 5)


if __name__ == '__main__':
    unittest.main()
