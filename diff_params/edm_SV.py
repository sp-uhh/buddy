import torch
import numpy as np

import utils.training_utils as utils
from diff_params.edm import EDM


class EDM_SV(EDM):
    """
        Definition of the diffusion parameterization, following ( Karras et al., "Elucidating...", 2022). 
        This includes only the utilities needed for training, not for sampling.
    """

    def __init__(self,
        type,
        sde_hp):

        super().__init__(type, sde_hp)

        self.sigma_data_sampling=sde_hp.sigma_data_sampling

    def denoiser(self, xn , net, t, *args, **kwargs):
        """
        This method does the whole denoising step, which implies applying the model and the preconditioning
        Args:
            x (Tensor): shape: (B,1,T) Intermediate noisy latent to denoise
            model (nn.Module): Model of the denoiser
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)

        #Eloi: I trained that singing voice model normalizing the data, so sigma_data=1. This was useful as I trained with different datasets. However, I think it is better to use the same schedule as with speech, so I decided to patch this
        """
        print(xn.std(), t)

        t=t/self.sigma_data_sampling
        xn=xn/self.sigma_data_sampling
        print("fixed",xn.std(), t)

        sigma = self._std(t).unsqueeze(-1)
        sigma = sigma.view(*sigma.size(), *(1,)*(xn.ndim - sigma.ndim))

        cskip = self.cskip(sigma)
        cout = self.cout(sigma)
        cin = self.cin(sigma)
        cnoise = self.cnoise(sigma.squeeze())

        #check if cnoise is a scalar, if so, repeat it
        if len(cnoise.shape) == 0:
            cnoise = cnoise.repeat(xn.shape[0],)
        else:
            cnoise = cnoise.view(xn.shape[0],)


        res= cskip * xn + cout * net(cin * xn, cnoise)  #this will crash because of broadcasting problems, debug later!

        #high-pass filter to accoint for the cropped DC component
        res=net.CQTransform.apply_hpf_DC(res)

        return res*self.sigma_data_sampling
