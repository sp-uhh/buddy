import torch
import numpy as np

import utils.training_utils as utils
from diff_params.shared import SDE

class VE(SDE):
    """
        Definition of the diffusion parameterization, following the VarianceExploding scheme from Song et al. ("Score-based Generative...", 2021)
        reparameterized as in ( Karras et al., "Elucidating...", 2022). 
        This includes only the utilities needed for training, not for sampling.
    """

    def __init__(self,
        type,
        sde_hp):

        super().__init__(type, sde_hp)

        self.sigma_min = self.sde_hp.sigma_min
        self.sigma_max = self.sde_hp.sigma_max
        self.t_eps = self.sde_hp.t_eps

    def sample_time_training(self,N):
        """
        For training, getting t according to a similar criteria as sampling.
        Args:
            N (int): batch size
        """
        a = torch.minimum(torch.rand(N), self.t_eps*torch.ones(N))
        t = self.sigma_min**2 * (self.sigma_max / self.sigma_min)**(2*a)
        return t

    def sample_prior(self,shape):
        """
        Just sample some gaussian noise, nothing more
        Args:
            shape (tuple): shape of the noise to sample, something like (B,T)
        """
        n = torch.randn(shape)
        return n

    def cskip(self, *ignored_args):
        """
        Just one of the preconditioning parameters
        
        """
        return 1.

    def cout(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return sigma

    def cin(self, *ignored_args):
        """
        Just one of the preconditioning parameters
        """
        return 1.

    def cnoise(self, sigma):
        """
        preconditioning of the noise embedding
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (1/2)*torch.log(sigma)

    def lambda_w(self, sigma):
        """
        Score matching loss weighting
        """
        return 1 / sigma**2

    def Tweedie2score(self, tweedie, xt, t, *args, **kwargs):
        return (tweedie - self._mean(xt, t)) / self._std(t)**2

    def score2Tweedie(self, score, xt, t, *args, **kwargs):
        return self._std(t)**2 * score + self._mean(xt, t)

    def _mean(self, x, t):
        return x
    
    def _std(self, t):
        return torch.sqrt(t)
    
    def _ode_integrand(self, x, t, score):
        return -.5 * score # Because look at equation 209 in Karras et al. EDM. since sigma = sqrt(t) actually the ODE drift term is dsigma * sigma = 1/2