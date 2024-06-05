import utils.reverb_utils as reverb_utils
import numpy as np
import torch.nn as nn
import torch
import torchcde
from testing.operators.shared import Operator

class SubbandFiltering(Operator):

    def __init__(self, op_hp, sample_rate):
        """
        Create a subband filter operator
        Informed scenarion, entire subband filter is known
        """
        super().__init__()
        self.H = None
        self.sample_rate = sample_rate    

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.op_hp = op_hp

        self.n_fft = op_hp.NFFT
        self.win_length = op_hp.win_length
        assert self.n_fft >= self.win_length, "n_fft must be greater than 2*win_length to avoid temporal aliasing"
        self.hop_length = op_hp.hop
        w = op_hp.window
        if w == "hann":
            self.window = torch.hann_window(self.win_length, device=self.device)
            assert self.hop_length <= self.win_length/4, "hop length must be less than 1/4 of win_length to avoid temporal aliasing" #(it should work with 1/2, but somehow it doesn't)
        else:
            raise NotImplementedError("window type {} not implemented".format(w))

        self.window_padded = torch.nn.functional.pad(self.window, (0,self.n_fft-self.win_length), mode='constant', value=0) #zero padding

        self.freqs = torch.fft.rfftfreq(self.n_fft, d=1/sample_rate).to(self.device)
        self.Nf = self.op_hp.Nf
        self.length_rir = self.hop_length * self.Nf
        self.time = torch.arange(self.Nf, dtype=torch.float32) / (self.sample_rate/self.hop_length)

    def apply_stft(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
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
            length_param = length + self.win_length//2

        X *= torch.sqrt(torch.sum(self.window_padded**2))
        x = self.istft(X, length=length_param)
        x = x[...,self.win_length//2:] #account for extra delay caused by centering the stft (see subband_filtering())

        return x    

    def subband_filtering(self, X, H):
        pre_impulse_frames = int((self.win_length//self.hop_length)/2) - 1

        H_filt = torch.flip(H,dims=[-1]).unsqueeze(1) #flip the filter and add a channel dimension  
        X = torch.nn.functional.pad(X, (H_filt.shape[-1]-1-pre_impulse_frames, pre_impulse_frames)) #centering the RIR where the hann window with the direct path at the center is at the origin
        Y_subband = torch.nn.functional.conv1d(X, H_filt, groups=H_filt.shape[0])

        return Y_subband

    def istft(self, X, length=None):
        return torch.istft(X, self.n_fft,hop_length=self.hop_length, win_length=self.n_fft, window=self.window_padded, onesided=True, center=True, normalized=False, return_complex=False, length=length)

    def stft(self, x):
        return torch.stft(x,  self.n_fft, hop_length=self.hop_length,win_length=self.n_fft, window=self.window_padded,  center=True, onesided=True, return_complex=True,normalized=False, pad_mode='constant')

    def degradation(self, x,mode="waveform", H=None, detach_operator=False):
        init_shape = x.shape
        X = self.apply_stft(x)

        if H is None:
            assert self.H is not None, "filter is not initialized"
            H = self.H
        if detach_operator:
            H = H.detach()

        Y_subband = self.subband_filtering(X, H)

        if mode == "waveform":
            y_subband = self.apply_istft(Y_subband, length=init_shape[-1])
            if len(init_shape) == 1:
                y_subband = y_subband.squeeze(0) #remove batch dimension
            return y_subband

        elif mode == "STFT":
            return Y_subband

    def get_time_RIR(self, excitation=None, H=None):
        if excitation is None:
            x = torch.zeros(int(self.length_rir+1024)).to(torch.float32).to(self.device)
            x[0] = 1
        else:
            x = torch.Tensor(excitation).to(self.device)

        if H is None:
            return self.degradation(x)
        else:
            return self.degradation(x, H=H)

    def update_H(self, rir=None, H=None):
        if rir is not None:
            H = self.stft(rir)
            H *= (8)/(self.win_length/(self.hop_length)) # scaling factor (hard coded for hann window)
            H = H[:,1:] #discard first sample to avoid divide by zero (it should have zero energy anyway)
            if self.op_hp.Nf > H.shape[-1]: #if we have speficied a larger number of frames, pad with zeros
                H = torch.cat((H,torch.zeros((H.shape[0],(self.op_hp.Nf-H.shape[-1]))).to(H.device)),-1)
            else: #otherwise, truncate
                H = H[...,0:self.op_hp.Nf]
            self.H = H
            print("updating H with time-domain rir")

        elif H is not None:
            self.H = H
            print("updating H with specified STFT H")

        else:
            raise ValueError("Either rir or H must be specified. This is the informed scenario, so we need to know the filter")

        assert self.H.shape[0] == self.n_fft//2+1, "H.shape: {}, n_fft//2 +1: {}".format(self.H.shape, self.n_fft//2+1)
        assert self.H.shape[1] == self.Nf, "H.shape: {}, Nf: {}".format(self.H.shape, self.Nf)

        return





class BlindSubbandFiltering(SubbandFiltering):

    def __init__(self, op_hp, sample_rate, magnitude_distance=True, H_cplx=False):
        """
        Create a subband filter operator
        Blind scenario, where we parameterize the subband filters with exponential decays
        """
        super().__init__(op_hp, sample_rate)

        self.Amin = self.op_hp.Amin
        self.Amax = self.op_hp.Amax
        self.EQ_freqs = torch.Tensor(self.op_hp.EQ_freqs).to(self.device)
        self.fix_EQ_extremes = self.op_hp.fix_EQ_extremes

        if self.fix_EQ_extremes:
            self.num_bands = len(self.EQ_freqs)-2
        else:
            self.num_bands = len(self.EQ_freqs)

        if self.op_hp.init_single_value:
            T60_breakpoints_init = [self.num_bands*[T60] for T60 in op_hp.init_params.T60_breakpoints]
            multiexp_weighting = [self.num_bands*[weight] for weight in  op_hp.init_params.multiexp_weighting]
        else:
            T60_breakpoints_init = op_hp.init_params.T60_breakpoints
            multiexp_weighting = op_hp.init_params.multiexp_weighting

        T60_breakpoints_init = torch.Tensor(T60_breakpoints_init).to(self.device)
        decay_breakpoints = 6.908/(T60_breakpoints_init*(self.sample_rate/op_hp.hop))
        self.num_exponentials = decay_breakpoints.shape[0]
        multiexp_weighting = torch.Tensor(multiexp_weighting).to(self.device)

        assert len(multiexp_weighting) == self.num_exponentials, "multiexp_weighting must have the same length as T60_breakpoints"

        if self.fix_EQ_extremes:
            assert T60_breakpoints_init.shape[-1] == (len(self.EQ_freqs)-2), "T60_breakpoints must have the same length as EQ_freqs-2"
            assert multiexp_weighting.shape[1] == (len(self.EQ_freqs)-2), "multiexp_weighting must have the same length as EQ_freqs-2"
        else:
            assert T60_breakpoints_init.shape[-1] == (len(self.EQ_freqs)), "T60_breakpoints must have the same length as EQ_freqs"
            assert multiexp_weighting.shape[1] == (len(self.EQ_freqs)), "multiexp_weighting must have the same length as EQ_freqs"

        self.params_decay = torch.nn.Parameter(decay_breakpoints)
        self.params_decay_weighting = torch.nn.Parameter(multiexp_weighting)  

        self.max_decay = 6.908/(op_hp.T60min*(self.sample_rate/self.hop_length))
        self.min_decay = 6.908/(op_hp.T60max*(self.sample_rate/self.hop_length))

        with torch.no_grad():
            self.phases = torch.rand((self.n_fft//2+1, self.Nf), dtype=torch.float32).to(self.device)*2*np.pi-np.pi
        self.phases = torch.nn.Parameter(self.phases, requires_grad=True)

        self.params = [self.params_decay, self.params_decay_weighting]
        self.params_phases = [self.phases]

        self.fix_direct_path = self.op_hp.fix_direct_path
        self.compute_direct_path_mag_correction()

        if self.op_hp.init_phases == "random_coherent":
            self.update_H(use_noise=True)
        elif self.op_hp.init_phases == "random":
            self.update_H()
        else:
            raise NotImplementedError("This is not implemented yet")

    def compute_direct_path_mag_correction(self):
        h = torch.zeros((self.length_rir,)).to(self.device)
        h[0] = 1*(self.win_length/(self.hop_length*2))
        H = self.stft(h)
        self.direct_path_mag_correction = H[:,1:].abs()

    def correct_OLA(self, A, inverse=False):
        K = int(self.win_length/(self.hop_length)-1)
        win_sum = torch.sum(self.window)
        for k in range(0, K):
            extra_correction=0
            if inverse:
                A[:,k] *= win_sum/torch.sum(self.window[int((K-k+extra_correction)*self.hop_length):])
            else:
                A[:,k] /= win_sum/torch.sum(self.window[int((K-k+extra_correction)*self.hop_length):])

        return A

    def design_subband_filter(self):
        Nf = len(self.time)
        decay_breakpoints = torch.exp(self.params[0])
        weights = self.params[1]

        decay_matrix = torch.zeros((len(self.EQ_freqs), Nf), device=self.device)
        decay_matrix[1:-1] = (weights.unsqueeze(-1)*decay_breakpoints.unsqueeze(-1)**(-torch.arange(0, Nf).to(decay_breakpoints.device).unsqueeze(0).unsqueeze(0).float())).sum(0)
        decay_matrix = torch.log(decay_matrix.transpose(0,1)+1e-6)
    
        coeffs = torchcde.linear_interpolation_coeffs(decay_matrix.unsqueeze(-1))
        spline = torchcde.LinearInterpolation(coeffs, t=self.EQ_freqs.to(torch.float32))
        H2 = spline.evaluate(self.freqs)
        H2 = torch.exp(H2.squeeze(-1).transpose(0,1))

        assert (torch.isnan(H2).any()==False), f"decay is Nan"
        return H2

    def design_filter(self, correct_OLA=True):
        A = self.design_subband_filter()+1e-6

        if correct_OLA:
            A = self.correct_OLA(A)
        if self.fix_direct_path:
            A += self.direct_path_mag_correction

        assert A.shape[0] == self.n_fft//2+1, "A.shape: {}, n_fft//2 +1: {}".format(A.shape, self.n_fft//2+1)
        assert A.shape[1] == self.op_hp.Nf, "A.shape: {}, Nf: {}".format(A.shape, self.op_hp.Nf)
        return A

    def get_noise(self, noise=None):
        if noise is None:
            noise=torch.randn((self.length_rir,)).to(self.device)

        N = self.stft(noise)/torch.sqrt(torch.sum(self.window_padded**2))
        N = N[:,1:]
        return N

    def update_H(self, rir=None, H=None, use_noise=False, noise=None, phases=None):
        if rir is not None:
            super().update_H(rir=rir)
        elif H is not None:
            super().update_H(H=H)
        else:
            A = self.design_filter()
            if use_noise:
                #initialize the phases with a noise signal. This way they are random but coherent (not sure if this is a good idea)
                N = self.get_noise(noise)
                self.H = A*torch.exp(1j*N.angle())
                self.H = self.cons(self.H, length=self.length_rir)

                self.params_phases[0] = torch.angle(self.H).detach()

            elif phases is not None:
                self.params_phases[0] = phases
                self.H = A*torch.exp(1j*phases)
                self.H = self.cons(self.H, length=self.length_rir)
            else:
                self.H = A*torch.exp(1j*self.params_phases[0])
                self.H = self.cons(self.H, length=self.length_rir)

        assert self.H.shape[0]==self.n_fft//2+1, "H.shape: {}, n_fft//2 +1: {}".format(self.H.shape, self.n_fft//2+1)
        assert self.H.shape[1]==self.Nf, "H.shape: {}, Nf: {}".format(self.H.shape, self.Nf)

    def update_params(self, params_dict):
        T60s = torch.Tensor(params_dict.T60_breakpoints).to(self.device)
        multiexp_weighting = torch.Tensor(params_dict.multiexp_weighting).to(self.device)
        decays = 6.908/(T60s*(self.sample_rate/self.hop_length))

        assert len(multiexp_weighting)==len(T60s), "multiexp_weighting must have the same length as T60_breakpoints"
        self.num_exponentials=len(T60s)

        self.params[0] = torch.nn.Parameter(decays, requires_grad=True)
        self.params[1] = torch.nn.Parameter(multiexp_weighting, requires_grad=True)

    def project_params(self):
        """
        Limit the parameters to the range of valid values
        """
        for i in range(len(self.params)):
            self.params[i].detach_()

        if self.op_hp.clamp_decay:
            for i in range(self.params[0].shape[0]):
                for k in range(self.params[0].shape[1]):
                    max = self.max_decay
                    if self.op_hp.strictly_decreasing_decay:
                        if k == 0:
                            min = self.min_decay
                        else:
                            min = self.params[0][i][k-1]
                    else:
                        min = self.min_decay

                    if i == 0:
                        self.params[0][i][k] = torch.clamp(self.params[0][i][k], min=min, max=max)
                    else:
                        if self.op_hp.enforce_long_decay_in_second_exponential:
                            max = torch.min(torch.Tensor([self.params[0][0][k]/1.01,max])).to(self.device)

                        self.params[0][i][k]=torch.clamp(self.params[0][i][k],min=min, max=max)

        for k in range(0,len(self.params[1][0])):
            self.params[1][0][k] = torch.clamp(self.params[1][0][k], min=10**(self.Amin/20), max=10**(self.Amax/20)) #main exponential must have at least 50% of the weight
            for i in range(1,len(self.params[1])):
                self.params[1][i][k] = torch.clamp(self.params[1][i][k], min=10**(self.Amin/20), max=self.params[1][0][k])

        assert (torch.isnan(self.params[0]).any()==False), f"decay is Nan"
        assert (torch.isnan(self.params[1]).any()==False), f"weights is Nan"
    
    def cons(self, X, length=None):
        L=X.shape[-1]
        X=torch.nn.functional.pad(X, (1,1))
        h=self.istft(X,length=length)
        #padding to make it fit
        h=torch.nn.functional.pad(h, (0,self.hop_length))

        if self.op_hp.minimum_phase:
            h = reverb_utils.minimum_phase_version(h)
            #if self.fix_direct_path:
            #    h[0]=1*(self.win_length/(self.hop_length*2)) #fix the direct path to 1 (hard coded for hann window)
        if self.fix_direct_path:
            #if h[0]>0:
            h[0]=1*(self.win_length/(self.hop_length*2)) #fix the direct path to 1 (hard coded for hann window)
            #else:
            #h[0]=-1*(self.win_length/(self.hop_length*2)) #fix the direct path to 1 (hard coded for hann window)
        X_rec= self.stft(h)
        X_rec=X_rec[:,1:-1]
        return X_rec[...,:L]
