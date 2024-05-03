
from tqdm import tqdm
import torch
import abc

class Sampler():

    def __init__(self, model, diff_params, args):

        self.model = model.eval() #is it ok to do this here?

        self.diff_params = diff_params #same as training, useful if we need to apply a wrapper or something
        self.args=args

        if self.args.tester.sampling_params.same_as_training:
            self.sde_hp = diff_params.sde_hp
        else:
            self.sde_hp = self.args.tester.sampling_params.sde_hp

        self.T = self.args.tester.sampling_params.T
        self.step_counter = 0

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict_unconditional(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict_conditional(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        pass

    def create_schedule(self, sigma_min=None, sigma_max=None, rho=None, T=None):
        """
        EDM schedule by default
        """
        if sigma_min is None:
            sigma_min = self.sde_hp.sigma_min
        if sigma_max is None:
            sigma_max = self.sde_hp.sigma_max
        if rho is None:
            rho = self.sde_hp.rho
        if T is None:
            T=self.T

        if self.args.tester.sampling_params.schedule == "edm":
            a = torch.arange(0, T+1)
            t = (sigma_max**(1/rho) + a/(T-1) *(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
            t[-1] = 0
            return t
        
        elif self.args.tester.sampling_params == "song":
            eps = 0. if not "t_eps" in self.args.tester.diff_params.keys() else self.args.tester.diff_params.t_eps
            a = torch.arange(eps, T+1)
            t = sigma_min**2 * (sigma_max / sigma_min)**(2*a)
            t[-1] = 0

        else:
            raise NotImplementedError(f"schedule {self.args.tester.posterior_sampling.RED.schedule} not implemented")

    def Tweedie2score(self, tweedie, xt, t):
        return self.diff_params.Tweedie2score(tweedie, xt, t)

    def get_Tweedie_estimate(self, x, t_i):
        x_hat = self.diff_params.denoiser(x.unsqueeze(1), self.model, t_i).squeeze(1)
        return x_hat

class NoSampler(Sampler):

    def predict(self, *args, **kwargs):
        return None

    def predict_unconditional(self, *args, **kwargs):
        return None

    def predict_conditional(self, *args, **kwargs):
        return None

    def step(self, *args, **kwargs):
        return None
