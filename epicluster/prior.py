"""Prior distribution of clusterings.
"""

import math
import scipy.special
import scipy.optimize
import epicluster as ec


class RestrictedPYEPPF:
    def __init__(self):
        super().__init__()

    def find_prior_hyperparam(self, n, theta=0.0, num_blocks=1.5):
        """Find the value of sigma corresponding to a given theta and expected
        clusters.
        """
        def f(sigma):
            return -num_blocks + math.exp(ec.log_poch(theta + sigma, n) - math.log(sigma) - ec.log_poch(theta+1, n-1)) - theta/sigma

        x = scipy.optimize.root_scalar(
            f, x0=0.5, bracket=[1e-10, 1-1e-10]).root

        return x

    def __call__(self, assignments, sigma, theta):
        """Evaluate the Prior Log pdf.

        Parameters
        ----------
        assignments : list of int
            List giving the current block assignment for each time point
        sigma : int
            Discount prior hyperparameter
        theta : int
            Strength prior hyperparameter

        Returns
        -------
        float
            The log of the current value of the prior evaluated at the block
            assignments
        """
        n = len(assignments)  # number of time points
        k = len(set(assignments))  # number of blocks

        # Start calculating the terms of the prior in log space
        p = math.lgamma(n+1) - math.lgamma(k+1)

        prod1 = 0
        for i in range(k-1):
            prod1 += math.log(theta + (i+1) * sigma)
        p += prod1

        p -= ec.log_poch(theta+1, n-1)

        prod2 = 0
        for j in range(k):
            n_j = assignments.count(j)
            prod2 += ec.log_poch(1-sigma, n_j-1) - math.lgamma(n_j+1)

        p += prod2

        return p
