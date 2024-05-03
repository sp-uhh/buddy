import utils.reverb_utils as reverb_utils
import torch.nn as nn
import torch
from testing.operators.shared import Operator
import utils.blind_bwe_utils as blind_bwe_utils

class BABE_LPF_Operator(Operator):
    def __init__(self, op_hp, sample_rate):
        """
        Create a placeholder for a convolution in the time domain
        """
        super().__init__()
        self.params=None

        self.op_hp=op_hp
        #we might not even need the sample rate here, but let's keep it for now
        self.freqs=torch.fft.rfftfreq(op_hp.NFFT, d=1/sample_rate)
        self.freqs=torch.nn.Parameter(self.freqs, requires_grad=False)

        self.params=torch.Tensor([op_hp.initial_conditions.fc,op_hp.initial_conditions.A])
        self.params=torch.nn.Parameter(self.params)

        #these parameters might be used for blind optimization
        self.Amin=op_hp.Amin
        self.Amax=op_hp.Amax
        self.fcmin=op_hp.fcmin
        self.fcmax=op_hp.fcmax

    def degradation(self, x):
        assert self.params is not None, "filter is not initialized"

        #I use fast connvolution because it is much faster than the standard convolution. I was running into some issues with the standard convolution
        # print("x.shape", x.shape, "self.params.shape", self.params.shape)

        H=blind_bwe_utils.design_filter_BABE(self.params, self.freqs)
        return blind_bwe_utils.apply_filter_BABE(x, H,self.op_hp.NFFT)

    def update_params(self, filter_params, **ignored_kwargs):
        assert len(filter_params)==2, "filter_params should be a tuple of length 2"
        assert len(filter_params[0])==len(filter_params[1]), "filter_params should be a tuple of length 2, where the first element is the fc and the second element is the A, both of the same length"
        filter=torch.Tensor([filter_params[0], filter_params[1]]).to(self.params.device)
        self.params.data=filter

    def prepare_optimization(self, x_den, y):
        """
        Some preprocessing for optimizing the parameters. I just compute the STFT separately to save a bit of computation
        """
        Xden=blind_bwe_utils.apply_stft(x_den, self.op_hp.NFFT)
        Y=blind_bwe_utils.apply_stft(y, self.op_hp.NFFT)
        return Xden, Y
        
    def constrain_params(self):
        """
        Limit the parameters to the range of valid values
        """
        if self.op_hp.optim.clamp_fc:
                    self.params[0,0]=torch.clamp(self.params[0,0],min=self.fcmin,max=self.fcmax)
                    for k in range(1,len(self.params[0])):
                        self.params[0,k]=torch.clamp(self.params[0,k],min=self.params[0,k-1]+1,max=self.fcmax)
        if self.op_hp.optim.clamp_A:
                    self.params[1,0]=torch.clamp(self.params[1,0],min=self.Amin,max=-1 if self.op_hp.optim.constrain_only_negative_A else self.Amax)
                    for k in range(1,len(self.params[0])):
                        self.params[1,k]=torch.clamp(self.params[1,k],min=self.Amin,max=self.params[1,k-1]-1 if self.op_hp.optim.constrain_only_negative_A else self.Amax)

    def optim_fwd(self, Xden, Y):
        """
        Xden: STFT of denoised estimate
        y: observations
        params: parameters of the degradation model (fc, A)
        """
        #print("before design filter", self.params)
        H=blind_bwe_utils.design_filter_BABE(self.params, self.freqs)
        return blind_bwe_utils.apply_filter_and_norm_STFTmag_fweighted(Xden, Y, H, self.op_hp.optim.freq_weighting_filter)

class LPFOperator():
    def __init__(self, args, device) -> None:
        self.args=args
        self.device=device

        self.params=torch.Tensor([self.args.tester.blind_bwe.initial_conditions.fc, self.args.tester.blind_bwe.initial_conditions.A]).to(device)
        self.params=torch.nn.Parameter(self.params)


        self.optimizer=torch.optim.Adam([self.params], lr=self.args.tester.blind_bwe.lr_filter) #self.mu=torch.Tensor([self.args.tester.blind_bwe.optimization.mu[0], self.args.tester.blind_bwe.optimization.mu[1]])

        if len(self.params.shape)==1:
            self.params.unsqueeze_(1)
        print(self.params.shape)
        self.shape_params=self.params.shape #fc and A

        self.freqs=torch.fft.rfftfreq(self.args.tester.blind_bwe.NFFT, d=1/self.args.exp.sample_rate).to(self.device)
        #self.degradation=lambda x, params: self.apply_filter_fcA(x,  params)

        self.fcmin=self.args.tester.blind_bwe.fcmin
        if self.args.tester.blind_bwe.fcmax =="nyquist":
                self.fcmax=self.args.exp.sample_rate//2
        else:
                self.fcmax=self.args.tester.blind_bwe.fcmax
        self.Amin=self.args.tester.blind_bwe.Amin
        self.Amax=self.args.tester.blind_bwe.Amax

        self.tol=self.args.tester.blind_bwe.optimization.tol


    def degradation(self, x):
        return self.apply_filter_fcA(x)

    def apply_filter_fcA(self, x):
        H=blind_bwe_utils.design_filter(self.params[0], self.params[1], self.freqs)
        return blind_bwe_utils.apply_filter(x, H,self.args.tester.blind_bwe.NFFT)

    def stop(self, prev_params):
        if (torch.abs(self.params[0]-prev_params[0]).mean()<self.tol[0]) and (torch.abs(self.params[1]-prev_params[1]).mean()<self.tol[1]):
            return True
        else:
            return False

    def limit_params(self):
        if self.args.tester.blind_bwe.optimization.clamp_fc:
                    self.params[0,0]=torch.clamp(self.params[0,0],min=self.fcmin,max=self.fcmax)
                    for k in range(1,len(self.params[0])):
                        self.params[0,k]=torch.clamp(self.params[0,k],min=self.params[0,k-1]+1,max=self.fcmax)
        if self.args.tester.blind_bwe.optimization.clamp_A:
                    self.params[1,0]=torch.clamp(self.params[1,0],min=self.Amin,max=-1 if self.args.tester.blind_bwe.optimization.only_negative_A else self.Amax)
                    for k in range(1,len(self.params[0])):
                        self.params[1,k]=torch.clamp(self.params[1,k],min=self.Amin,max=self.params[1,k-1]-1 if self.args.tester.blind_bwe.optimization.only_negative_A else self.Amax)
    def optimizer_func(self, Xden, Y):
        """
        Xden: STFT of denoised estimate
        y: observations
        params: parameters of the degradation model (fc, A)
        """

        #print("before design filter", self.params)
        H=blind_bwe_utils.design_filter(self.params[0],self.params[1], self.freqs)
        return blind_bwe_utils.apply_filter_and_norm_STFTmag_fweighted(Xden, Y, H, self.args.tester.posterior_sampling.freq_weighting_filter)
