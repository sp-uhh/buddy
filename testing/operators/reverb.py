import utils.reverb_utils as reverb_utils
import torch.nn as nn
import torch
import utils.log as logging_utils
from testing.operators.shared import Operator
import torch

class RIROperator(Operator):
    def __init__(self, op_hp, time_kernel_size=10, sample_rate=16000):
        """
        Create a placeholder for a convolution in the time domain
        """
        super().__init__()
        self.time_kernel_size = time_kernel_size
        self.params=None

        ### Just for computing STFT-based losses ###
        self.sample_rate=sample_rate
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.n_fft=op_hp.NFFT
        self.win_length=op_hp.win_length
        self.hop_length=op_hp.hop
        w=op_hp.window
        if w=="hann":
            self.window=torch.hann_window(self.win_length, device=self.device)
            #assert hop is less than 1/4 of win_length, otherwise OLA will not work (it should work with 1/2, but somehow it doesn't)
            assert self.hop_length<=self.win_length/4, "hop length must be less than 1/4 of win_length to avoid temporal aliasing"
        else:
            raise NotImplementedError("window type {} not implemented".format(w))
        self.window_padded=torch.nn.functional.pad(self.window, (0,self.n_fft-self.win_length), mode='constant', value=0) #zero padding
        self.freqs=torch.fft.rfftfreq(self.n_fft, d=1/sample_rate).to(self.device)

    def degradation(self, x, rm_delay=False, **ignored_kwargs):
        assert self.params is not None, "filter is None"
        return reverb_utils.fast_apply_RIR(x, self.params, rm_delay=rm_delay)

    def update_params(self, k, **ignored_kwargs):
        if self.params is None:
            self.params=torch.nn.Parameter(k, requires_grad=False)
        else:
            self.params.data=k

    def optim_fwd(self, Xden, Y):
        """
        Compute the reconstruction loss between our current estimate and the observation, knowing the parameterized forward model 
        """
        self.rec_loss_fn = lambda x, x_hat: torch.sum((x-x_hat)**2)
        Y_estimate = self.degradation(Xden)
        return self.rec_loss_fn(Y_estimate, Y)


    ### Just for computing STFT-based losses ###

    def apply_stft(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape)==2:
            pass
        else:
            raise ValueError("x must have shape (batch, samples) or (samples)")

        xpadded = torch.nn.functional.pad(x, (0,self.win_length)) #bad things happen at the end if you don't pad (because of the centering of torch.sftft and the padded window, I think)
        X = self.stft(xpadded)/torch.sqrt(torch.sum(self.window_padded**2))

        return X

    def apply_istft(self, X, length=None):
        if length is None:
            print("Warning: length is None, istft may crash")
            length_param = None
        else:
            length_param = length+self.win_length//2

        X *= torch.sqrt(torch.sum(self.window_padded**2))
        x = self.istft(X, length=length_param)
        x = x[...,self.win_length//2:] #account for extra delay caused by centering the stft

        return x    

    def istft(self, X, length=None):
        return torch.istft(X, self.n_fft,hop_length=self.hop_length, win_length=self.n_fft, window=self.window_padded, onesided=True, center=True, normalized=False, return_complex=False, length=length)

    def stft(self, x):
        return torch.stft(x,  self.n_fft, hop_length=self.hop_length,win_length=self.n_fft, window=self.window_padded,  center=True, onesided=True, return_complex=True,normalized=False, pad_mode='constant')

    def get_time_RIR(self):
        return self.params
    