"""Runs MCMC sampling on models.
"""

import copy
import numpy as np
import pints


class MCMCSampler:
    """Class for running mcmc inference for the posterior.
    """
    def __init__(self, model, num_chains):
        self.models = []
        for _ in range(num_chains):
            self.models.append(copy.deepcopy(model))

    def run_mcmc(self,
                 num_mcmc_samples=0,
                 Rhat_thresh=0,
                 progress=False,
                 max_mcmc=10000):
        """Run one MCMC step to generate samples from the posterior.

        Parameters
        ----------
        num_mcmc_samples : int
            The total number of MCMC samples to run
        progress : bool, optional (False)
            Whether or not to print iteration number

        Returns
        -------
        list
            MCMC chain of parameter values
        list
            MCMC chain of assignments to regimes
        list
            Number of regimes
        """
        if num_mcmc_samples == 0 and Rhat_thresh == 0:
            raise ValueError('Must provide stopping criteria')

        # Set maximum number of MCMC samplers if using Rhat
        if num_mcmc_samples == 0:
            num_mcmc_samples = max_mcmc

        T = len(self.models[0].assignments)
        if Rhat_thresh != 0:
            self.models[0].set_initial_blocks(T, 1)
            self.models[-1].set_initial_blocks(T, T)

        params_chain = []
        assign_chain = []
        blocks_chain = []

        for iter in range(num_mcmc_samples):
            params_chain.append([])
            assign_chain.append([])
            blocks_chain.append([])

            for model in self.models:
                model.run_mcmc_step()
                params_chain[-1].append(copy.deepcopy(model.change_params))
                assign_chain[-1].append(copy.deepcopy(model.assignments))
                blocks_chain[-1].append(len(set(model.assignments)))

            if Rhat_thresh != 0 and iter > 10 and iter%50 == 0:
                # Check if converged
                rhat = pints.rhat(np.asarray(blocks_chain).T[:, iter//2:])
                if progress:
                    print('Iter={}, Rhat={}'.format(iter, rhat))
                if rhat < Rhat_thresh:
                    print('Converged', iter, rhat)
                    break

        # return the first chain
        return [x[0] for x in params_chain], \
               [x[0] for x in assign_chain], \
               [len(set(x[0])) for x in assign_chain]
