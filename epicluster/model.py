"""Bayesian nonparametric model of changepoints.

It does not include any particular likelihood. Rather, the marginal likelihood
must be implemented by subclasses.
"""

import math
import copy
import random
import numpy as np
import epicluster as ec


class ChangepointProcess:
    """Change point process including MCMC proposals.

    Attributes
    ----------
    self.assignments : list of int
        List of cluster assignment indicators
    self.change_params : list of float
        Value of the parameter within each cluster

    Examples
    --------
    For two, equally sized clusters of three time points:
    self.assignments = [0, 0, 0, 1, 1, 1]
    self.change_params = [1.5, 0.5]
    """
    def __init__(self, hyper_sigma=0.5, hyper_theta=0):
        self.hyper_sigma = hyper_sigma
        self.hyper_theta = hyper_theta
        self.q = 0.5

    def set_initial_blocks(self, num_time_pts, num_blocks):
        """Initialize the block configuration.

        The blocks are uniformly spaced. The parameter value in each block is
        1.0

        The initial condition is saved to self.assignments and
        self.change_params

        Parameters
        ----------
        num_blocks : int
            Starting number of blocks. They will be of equal size.
        """
        z = list(range(num_blocks)) * (num_time_pts // num_blocks)
        z.sort()
        self.assignments = z.copy()[:num_time_pts]

        assert len(self.assignments) == num_time_pts

        phi = []
        for _ in range(len(set(self.assignments))):
            phi.append(1.0)

        self.change_params = copy.deepcopy(phi)

    def marginal_likelihood(self):
        """The marginal probability of the data conditional on assignments.

        The change parameters should be integrated out.
        """
        raise NotImplementedError

    def posterior(self):
        """Evaluate the product of the marginal likelihood and the prior over
        regime configurations.

        Returns
        -------
        float
            Log posterior value
        """
        p = self.marginal_likelihood()
        p += ec.RestrictedPYEPPF()(
            self.assignments, self.hyper_sigma, self.hyper_theta)
        return p

    def update_change_params(self):
        """Update the parameter values conditional on the current assignments.
        """
        raise NotImplementedError

    def run_mcmc_step(self, progress=False):
        """Run one MCMC step to generate samples from the posterior.
        """
        k = len(set(self.assignments))
        if progress:
            print(k)
            print(self.change_params)
            print('\n')

        # Randomly choose either split or merge
        if (k == 1 or random.random() < self.q) and k < len(self.cases):
            self._split_step()

        else:
            self._merge_step()

        # Recalculate k in case it changed in this iteration
        k = len(set(self.assignments))

        # Shuffle if possible
        if k > 1:
            self._shuffle_step()

        # Update parameters within each block
        self.update_change_params()

    def _split_step(self):
        """Propose a split, and accept or reject it.
        """
        k = len(set(self.assignments))
        z = copy.deepcopy(self.assignments)
        phi = copy.deepcopy(self.change_params)

        # Get all blocks with greater than 1 member
        splittable_blocks = \
            [block_idx for block_idx in set(z) if z.count(block_idx) > 1]

        # Choose a random one of those blocks
        j = random.choice(splittable_blocks)

        # Choose a random location within that block
        l = random.randint(1, z.count(j) - 1)

        # Build the proposed split vector
        j_time_idx = z.index(j)
        next_time_idx = j_time_idx + z.count(j)
        z_prop = z[:j_time_idx].copy()
        for _ in range(l):
            z_prop.append(j)
        for _ in range(z.count(j)-l):
            z_prop.append(j+1)
        z_prop += [x + 1 for x in z[next_time_idx:]]

        # Calculate acceptance ratio of the proposal
        p_old = self.posterior()

        self.assignments = z_prop.copy()

        p_prop = self.posterior()

        log_alpha = p_prop - p_old

        if k > 1:
            log_alpha += math.log(1-self.q) - math.log(self.q)
            ns = z.count(j)
            ngk = len(splittable_blocks)
            log_alpha += math.log(ngk * (ns - 1)) - math.log(k)

        elif k == 1:
            log_alpha += math.log(1-self.q) + math.log(len(self.cases)-1)

        cond = (math.log(random.random()) >= log_alpha)
        if cond:
            self.assignments = copy.deepcopy(z)

    def _merge_step(self):
        """Propose a merge, and accept or reject it.
        """
        k = len(set(self.assignments))
        z = copy.deepcopy(self.assignments)
        phi = copy.deepcopy(self.change_params)
        j = random.randint(0, k-2)

        # Build the proposed merged vector
        j_time_idx = z.index(j+1)
        z_prop = copy.deepcopy(z[:j_time_idx])
        z_prop += [x - 1 for x in z[j_time_idx:]]

        # Calculate acceptance ratio of the proposal
        p_old = self.posterior()
        self.assignments = z_prop.copy()
        p_prop = self.posterior()

        log_alpha = p_prop - p_old

        if k < len(self.cases):
            log_alpha += math.log(self.q) - math.log(1-self.q)
            ns = z.count(j)
            ns1 = z.count(j+1)
            ngk1 = len([block_idx for block_idx in set(z_prop)
                        if z_prop.count(block_idx) > 1])
            log_alpha += math.log(k-1) - math.log(ngk1 * (ns + ns1 - 1))

        elif k == len(self.cases):
            log_alpha += math.log(self.q) + math.log(len(self.cases)-1)

        cond = (math.log(random.random()) >= log_alpha)
        if cond:
            self.assignments = copy.deepcopy(z)

    def _shuffle_step(self):
        """Propose a shuffle, and accept or reject it.
        """
        k = len(set(self.assignments))
        z = copy.deepcopy(self.assignments)

        for _ in range(5):
            # Perform the shuffle step
            # Choose a random block, which is not the last
            i = random.randint(0, k-2)
            i_time_index = z.index(i)
            next_time_index = i_time_index + z.count(i) + z.count(i+1)

            ni = z.count(i)
            ni1 = z.count(i+1)

            # Choose a new point for the change point somewhere within the two
            # blocks
            j = random.randint(0, ni + ni1 - 2)

            # Build the proposed shuffled assignments
            z_prop = z[:i_time_index]

            for _ in range(j + 1):
                z_prop.append(i)

            for _ in range(ni+ni1-j - 1):
                z_prop.append(i+1)

            z_prop += z[next_time_index:]

            p_old = self.posterior()
            self.assignments = z_prop.copy()
            p_prop = self.posterior()

            if not math.isfinite(p_prop):
                self.assignments = z.copy()
                continue

            cond = (math.log(random.random()) >= p_prop - p_old)
            if cond:
                # Reject the proposal
                self.assignments = z.copy()
