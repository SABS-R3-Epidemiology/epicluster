"""Test the code in the module model.py.
"""

import math
import unittest
from unittest.mock import patch
import epicluster as ec


class TestModel(unittest.TestCase):

    def test_init(self):
        model = ec.ChangepointProcess(0.5, 0.6)
        self.assertEqual(model.hyper_sigma, 0.5)
        self.assertEqual(model.hyper_theta, 0.6)

    def test_set_initial_blocks(self):
        model = ec.ChangepointProcess(0.5, 0.6)
        model.set_initial_blocks(10, 2)
        self.assertEqual(model.assignments, [0,0,0,0,0,1,1,1,1,1])

        model.set_initial_blocks(10, 10)
        self.assertEqual(model.assignments, [0,1,2,3,4,5,6,7,8,9])

        # Check that all entries in change_params are equal to 1
        self.assertEqual(model.change_params[0], 1)
        self.assertEqual(len(set(model.change_params)), 1)

    def test_marginal_likelihood(self):
        model = ec.ChangepointProcess()
        with self.assertRaises(NotImplementedError):
            model.marginal_likelihood()

    @patch('epicluster.ChangepointProcess.marginal_likelihood')
    def test_posterior(self, mock_ml):
        mock_ml.return_value = 5.0
        model = ec.ChangepointProcess()
        model.assignments = [1, 2, 3]

        self.assertEqual(
            model.posterior(),
            5.0 + ec.RestrictedPYEPPF()(model.assignments,
                model.hyper_sigma, model.hyper_theta))

    def test_update_change_params(self):
        model = ec.ChangepointProcess()
        with self.assertRaises(NotImplementedError):
            model.update_change_params()

    @patch('epicluster.ChangepointProcess._split_step')
    @patch('epicluster.ChangepointProcess._merge_step')
    @patch('epicluster.ChangepointProcess._shuffle_step')
    @patch('epicluster.ChangepointProcess.update_change_params')
    def test_run_mcmc_step(self,
                           mock_change,
                           mock_shuffle,
                           mock_merge,
                           mock_split):
        model = ec.ChangepointProcess()
        model.assignments = [0, 0, 1, 1, 1]
        model.cases = [5, 6, 7, 8, 9]

        model.q = 1.0
        model.run_mcmc_step()

        # Check that the proposals were called
        mock_change.assert_called_once()
        mock_shuffle.assert_called_once()
        mock_split.assert_called_once()

        model.q = 0.0
        model.run_mcmc_step()
        # Check that the merge was called now that q is set to zero
        mock_merge.assert_called_once()


if __name__ == '__main__':
    unittest.main()
