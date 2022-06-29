"""Poisson renewal model.
"""

import math
import numpy as np
import copy
import scipy.special
import epicluster as ec


class PoissonModel(ec.ChangepointProcess):
    """Renewal model for local and imported cases using the Poisson
    distribution.
    """
    def __init__(self,
                 cases,
                 serial_interval,
                 imported_cases=None,
                 epsilon=1,
                 hyper_sigma=0.1,
                 hyper_theta=0,
                 prior_expected_clusters=None):
        """
        Parameters
        ----------
        cases : list of int
            Local cases, including historical cases prior to the inference
            interval. Historical cases should be equal in length to the
            supplied serial interval.
        serial_interval : list of float
            Discrete serial interval distribution
        imported_cases : list of int, optional
            Imported cases (those infected outside of the region)
        epislon : float, optional (1)
            Relative risk of onwards tranmission for imported cases compared
            to local cases
        hyper_sigma : float
            Hyperparameter sigma of the EPPF
        hyper_theta : float
            Hyperparameter theta of the EPPF
        prior_expected_clusters : float
            If supplied, chooses hyper_sigma such that the prior mean on number
            of clusters is equal to this value
        """
        super().__init__(hyper_sigma, hyper_theta)

        self.all_cases = cases
        self.imported_cases = imported_cases
        self.epsilon = epsilon
        self.serial_interval = serial_interval

        self.cases = copy.deepcopy(self.all_cases)[len(serial_interval):]
        self.set_initial_blocks(len(cases)-len(serial_interval), 1)

        self._calculate_lambdas()

        self.r_prior_alpha = 1.0
        self.r_prior_beta = 1/5.0

        if prior_expected_clusters is not None:
            self._set_sigma(prior_expected_clusters)

    def _set_sigma(self, expected_clusters):
        """Set sigma such that the prior mean is given by expected_clusters.
        """
        prior = ec.RestrictedPYEPPF()
        self.hyper_sigma = prior.find_prior_hyperparam(
            len(self.cases), num_blocks=expected_clusters)

    def _calculate_lambdas(self):
        """Calculate the tranmission potential for each day.
        """
        serial_interval = self.serial_interval
        past = len(serial_interval)
        lambdas = []
        for i in range(len(serial_interval), len(self.all_cases)):
            past_cases = copy.deepcopy(self.all_cases[i-past:i])
            if self.imported_cases is not None:
                past_cases = np.asarray(past_cases)
                past_cases += self.epsilon \
                    * np.asarray(self.imported_cases[i - past:i])

            lambdas.append(
                np.dot(past_cases, serial_interval[::-1][:len(past_cases)]))

        self.precalc_lambdas = copy.deepcopy(lambdas)

        self.precalc_ll_terms = []
        for c, l in zip(self.cases, self.precalc_lambdas):
            if l > 0:
                self.precalc_ll_terms.append(
                    c * math.log(l) - scipy.special.loggamma(c +1))
            else:
                self.precalc_ll_terms.append(0)

    def marginal_likelihood(self):
        serial_interval = self.serial_interval
        past = len(serial_interval)

        a = self.r_prior_alpha
        b = self.r_prior_beta

        mll = 0
        for block in set(self.assignments):
            block_start = self.assignments.index(block)
            block_end = block_start +self.assignments.count(block) -1

            cases_in_block = self.cases[block_start:block_end+1]
            lambdas_in_block = self.precalc_lambdas[block_start:block_end+1]

            mll += math.log(b**a) \
                  - scipy.special.loggamma(a) \
                  + scipy.special.loggamma(a + sum(cases_in_block)) \
                  + (-a - sum(cases_in_block)) * math.log(b + sum(lambdas_in_block))

            mll += sum(self.precalc_ll_terms[block_start:block_end+1])

        return mll

    def update_change_params(self):
        """Update change parameters within each block using Gibbs steps.
        """
        serial_interval = self.serial_interval
        past = len(serial_interval)

        a = self.r_prior_alpha
        b = self.r_prior_beta

        self.change_params = []

        for block in set(self.assignments):
            block_start = self.assignments.index(block)
            block_end = block_start + self.assignments.count(block) -1

            cases_in_block = self.cases[block_start:block_end+1]
            lambdas_in_block = self.precalc_lambdas[block_start:block_end+1]

            self.change_params.append(scipy.stats.gamma.rvs(
                a + sum(cases_in_block),
                scale=1/(b+sum(lambdas_in_block))))
