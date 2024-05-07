from tqdm import tqdm
import utils.log as utils_logging
import torch
import torchaudio
import numpy as np
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft
import wandb
import os
import utils.reverb_utils as reverb_utils
from utils.losses import get_loss

from testing.EulerHeunSampler import EulerHeunSampler

class EulerHeunSamplerDPS(EulerHeunSampler):
    """
        Euler Heun sampler for DPS 
        inverse problem solver
    """

    def __init__(self, model, diff_params, args):
        super().__init__(model, diff_params, args)
        self.zeta = self.args.tester.posterior_sampling.zeta

    def initialize_x(self, shape, device, schedule):
        if self.args.tester.posterior_sampling.warm_initialization.mode == "none":
            x = schedule[0]*torch.randn(shape).to(device)

        elif self.args.tester.posterior_sampling.warm_initialization.mode == "reverb_scaled":
            x = self.args.tester.posterior_sampling.warm_initialization.scaling_factor * self.y.clone() / self.y.std() + schedule[0] * torch.randn(shape).to(device)
        
        elif self.args.tester.posterior_sampling.warm_initialization.mode == "wpe_scaled":
            print("Processing WPE")
            stft_options = dict(size=512, shift=128)

            delay = self.args.tester.posterior_sampling.warm_initialization.wpe.delay
            iterations = self.args.tester.posterior_sampling.warm_initialization.wpe.iterations
            taps = self.args.tester.posterior_sampling.warm_initialization.wpe.taps
            
            Y = stft(self.y.cpu().numpy(), **stft_options)
            Y = Y.transpose(2, 0, 1)
            Z = wpe(
                Y,
                taps=taps,
                delay=delay,
                iterations=iterations,
                statistics_mode='full'
            ).transpose(1, 2, 0)
            x_pred = torch.from_numpy(istft(Z, size=stft_options['size'], shift=stft_options['shift'])).to(self.y.device).type(self.y.dtype)
            if x_pred.shape[-1] > self.y.shape[-1]:
                x_pred = x_pred[..., :self.y.shape[-1]]

            x_pred = self.args.tester.posterior_sampling.warm_initialization.scaling_factor * x_pred / x_pred.std()
            x = x_pred + schedule[0] * torch.randn(shape).to(device)

        else:
            raise NotImplementedError
        
        return x
    
    def get_likelihood_score(self, x_den, x, t):

        y_hat = self.operator.degradation(x_den, mode="waveform")
        rec = self.rec_loss(self.y, y_hat)
        rec_grads = torch.autograd.grad(outputs=rec, inputs=x)[0]

        # Normalize weighting parameter zeta
        normguide = torch.norm(rec_grads)/(self.args.exp.audio_len**0.5)
        return self.zeta / (normguide+1e-8) * rec_grads, rec
        
    def optimize_op(self, x_den, t):
        """
        Optimize the operator parameters
        """

        for _ in range(self.args.tester.posterior_sampling.blind_hp.op_updates_per_step):

            for k in range(len(self.operator.params)):
                self.operator.params[k].requires_grad=True
            for k in range(len(self.operator.params_phases)):
                self.operator.params_phases[k].requires_grad=True

            self.operator.update_H()

            # Reconstruction loss
            y_hat = self.operator.degradation(x_den, mode="waveform")
            if self.rec_loss_params is not None:
                rec_loss = self.rec_loss_params(self.y, y_hat)
                loss = rec_loss
                assert (torch.isnan(rec_loss).any()==False), f"rec_loss is Nan"
            else:
                loss = 0.

            # RIR noise regularization
            if self.RIR_noise_regularization_loss is not None:
                rir_time = self.operator.get_time_RIR()
                rir_noise = torch.randn_like(rir_time).to(x_den.device)
                t_op = max(min(t, self.args.tester.posterior_sampling.RIR_noise_regularization.crop_sigma_max), self.args.tester.posterior_sampling.RIR_noise_regularization.crop_sigma_min)
                rir_noisy = rir_time + t_op * rir_noise
                reg_loss = self.RIR_noise_regularization_loss(rir_time, rir_noisy.detach()) #detach gradients so that we do not backpropagate through the RIR operator
                loss += reg_loss

            assert (torch.isnan(loss).any()==False), f"loss is Nan"

            self.optimizer_operator.zero_grad()
            loss.backward()
            self.optimizer_operator.step()

            for p in self.operator.params:
                p.detach_()
            self.operator.project_params()
            for p in self.operator.params:
                p.requires_grad=True

    def step(self, x_i, t_i, t_iplus1, gamma_i, blind=False):

        x_hat, t_hat = self.stochastic_timestep(x_i, t_i, gamma_i)
        x_hat.requires_grad = True
        x_den = self.get_Tweedie_estimate(x_hat, t_hat)

        if blind:
            self.optimize_op(x_den.clone().detach(), t_hat)

        lh_score, rec_loss_value = self.get_likelihood_score(x_den, x_hat, t_hat)
        x_hat.detach_()

        # Rescale denoised speech estimate magnitude to constraint absolute magnitudes of RIR / speech estimate
        if self.args.tester.posterior_sampling.constraint_speech_magnitude.use:
            x_den = self.args.tester.posterior_sampling.constraint_speech_magnitude.speech_scaling / x_den.detach().std() * x_den #Match the sigma_data of dataset

        score = self.Tweedie2score(x_den, x_hat, t_hat)

        ode_integrand = self.diff_params._ode_integrand(x_hat, t_hat, score) + lh_score
        dt = t_iplus1 - t_hat

        if t_iplus1 !=0 and self.order == 2: #second order correction
            t_prime = t_iplus1
            x_prime = x_hat + dt * ode_integrand
            x_prime.requires_grad_(True)
            x_den = self.get_Tweedie_estimate(x_prime, t_prime)

            if blind:
                self.optimize_op(x_den.clone().detach(), t_prime)

            lh_score_next, rec_loss_value = self.get_likelihood_score(x_den, x_prime, t_prime)
            x_prime.detach_()

            score = self.Tweedie2score(x_den, x_prime, t_prime)

            ode_integrand_next = self.diff_params._ode_integrand(x_prime, t_prime, score) + lh_score_next
            ode_integrand_midpoint = .5 * (ode_integrand + ode_integrand_next)
            x_iplus1 = x_hat + dt * ode_integrand_midpoint
            
        else:
            x_iplus1 = x_hat + dt * ode_integrand

        return x_iplus1.detach_(), x_den.detach()

    def predict(
        self,
        shape, 
        device,
        blind=False
    ):
        # get the noise schedule
        t = self.create_schedule().to(device)

        # sample prior
        x = self.initialize_x(shape,device, t)

        # parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic
        gamma = self.get_gamma(t).to(device)

        for i in tqdm(range(0, self.T, 1)):
            self.step_counter=i
            x, x_den = self.step(x, t[i] , t[i+1], gamma[i], blind)
            
        return x_den.detach()
    
    def predict_unconditional(self, *args, **kwargs):
        raise ValueError("DPS not made for unconditional sampling")

    def predict_conditional(
        self,
        y,  #observations 
        operator, #degradation operator (assuming we define it in the tester)
        shape=None,
        blind=False,
        **kwargs
    ):

        self.operator = operator
        self.y = y
        self.rec_loss = get_loss(self.args.tester.posterior_sampling.rec_loss, operator=self.operator)

        if blind:
            self.rec_loss_params = get_loss(self.args.tester.posterior_sampling.rec_loss_params, operator=self.operator)
            self.optimizer_operator = torch.optim.Adam(self.operator.params + self.operator.params_phases, lr=self.args.tester.posterior_sampling.blind_hp.lr_op, weight_decay=self.args.tester.posterior_sampling.blind_hp.weight_decay, betas=(self.args.tester.posterior_sampling.blind_hp.beta1, self.args.tester.posterior_sampling.blind_hp.beta2))
            self.RIR_noise_regularization_loss = get_loss(self.args.tester.posterior_sampling.RIR_noise_regularization.loss, operator=self.operator)

        if shape is None:
            shape = y.shape

        return self.predict(shape, y.device, blind)
