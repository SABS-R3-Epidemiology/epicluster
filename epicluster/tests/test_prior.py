"""Test the code in the module prior.py.
"""

import math
import unittest
import epicluster as ec


class TestPrior(unittest.TestCase):

    def test_init(self):
        prior = ec.RestrictedPYEPPF()

    def test_call(self):
        prior = ec.RestrictedPYEPPF()

        ns = [1, 3, 5]
        n = sum(ns)
        k = len(ns)
        sigma = 0.45
        theta = 1.34

        expected = math.factorial(n) / math.factorial(k)
        for i in [1, 2]:
            expected *= (theta + i * sigma)
        expected /= math.exp(ec.log_poch(theta + 1, n-1))
        for j in [1, 2, 3]:
            expected *= (math.exp(ec.log_poch(1-sigma, ns[j-1]-1)) / math.factorial(ns[j-1]))

        zs = [0] * ns[0] + [1] * ns[1] + [2] * ns[2]
        self.assertAlmostEqual(math.log(expected), prior(zs, sigma, theta))

    def test_find_prior_hyperparam(self):
        def expected_clusters(sigma, theta, n):
            return math.exp(ec.log_poch(theta + sigma, n) - math.log(sigma) - ec.log_poch(theta+1, n-1)) - theta/sigma

        prior = ec.RestrictedPYEPPF()

        sigma = prior.find_prior_hyperparam(20)
        self.assertAlmostEqual(expected_clusters(sigma, 0.0, 20), 1.5)

        sigma = prior.find_prior_hyperparam(150)
        self.assertAlmostEqual(expected_clusters(sigma, 0.0, 150), 1.5)

        sigma = prior.find_prior_hyperparam(450)
        self.assertAlmostEqual(expected_clusters(sigma, 0.0, 450), 1.5)

        sigma = prior.find_prior_hyperparam(450, num_blocks=5)
        self.assertAlmostEqual(expected_clusters(sigma, 0.0, 450), 5)


if __name__ == '__main__':
    unittest.main()
