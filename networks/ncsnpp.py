# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

from .ncsnpp_utils import layers, layerspp, normalization
# from ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np
import einops


ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

def get_window(window_type, window_length):
	if window_type == 'sqrthann':
		return torch.sqrt(torch.hann_window(window_length, periodic=True))
	elif window_type == 'hann':
		return torch.hann_window(window_length, periodic=True)
	else:
		raise NotImplementedError(f"Window type {window_type} not implemented!")

class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, 
        nonlinearity = 'swish',
        nf = 128,
        ch_mult = (1, 2, 2, 2),
        num_res_blocks = 1,
        attn_resolutions = (0,),
        resamp_with_conv = True,
        time_conditional = True,
        fir = False,
        fir_kernel = [1, 3, 3, 1],
        skip_rescale = True,
        resblock_type = 'biggan',
        progressive = 'output_skip',
        progressive_input = 'input_skip',
        progressive_combine = 'sum',
        init_scale = 0.,
        fourier_scale = 16,
        image_size = 256,
        embedding_type = 'fourier',
        input_channels = 4,
        spatial_channels = 1,
        dropout = .0,
        centered = True,
        discriminative = False,
        **kwargs):
        super().__init__()

        self.FORCE_STFT_OUT = False

        self.act = act = get_act(nonlinearity)

        self.nf = nf = nf
        ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        dropout = dropout
        resamp_with_conv = resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]
        
        self.discriminative = discriminative
        if self.discriminative:
            # overwrite options that make no sense for a discriminative model
            time_conditional = False
            print("Running NCSN++ as discriminative backbone")
            input_channels = 2  # y.real, y.imag

        self.time_conditional = time_conditional  # noise-time_conditional
        self.centered = centered
        fir = fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        init_scale = init_scale
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)
        self.input_channels = input_channels
        self.spatial_channels = spatial_channels
        self.total_channels = self.input_channels * self.spatial_channels

        self.output_layer = nn.Conv2d(self.total_channels, 2*self.spatial_channels, 1)

        modules = []

        #######################
        ### MODULES NATURES ###
        #######################

        AttnBlock = functools.partial(layerspp.AttnBlockpp, 
            init_scale=init_scale, skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample, 
            with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample, fir=fir, 
                fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, 
                dropout=dropout, init_scale=init_scale, 
                skip_rescale=skip_rescale, temb_dim=(nf * 4 if time_conditional else None))
        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel, 
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=(nf * 4 if time_conditional else None))
        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        ######################
        ### TIME EMBEDDING ###
        ######################


        if time_conditional:
            if embedding_type == 'fourier':
                # Gaussian Fourier features embeddings.
                # assert config.training.continuous, "Fourier features are only used for continuous training."
                modules.append(layerspp.GaussianFourierProjection(
                    embedding_size=nf, scale=fourier_scale
                ))
                embed_dim = 2 * nf
            elif embedding_type == 'positional':
                embed_dim = nf
            else:
                raise ValueError(f'embedding type {embedding_type} unknown.')

            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        ##########################
        ### Downsampling block ###
        ##########################

        if progressive_input != 'none':
            input_pyramid_ch = self.total_channels

        modules.append(conv3x3(self.total_channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        
        ##########################
        ### Upsampling block ###
        ##########################

        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):  # +1 blocks in upsampling because of skip connection from combiner
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), 
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, self.total_channels, init_scale=init_scale))
                        pyramid_ch = self.total_channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, self.total_channels, bias=True, init_scale=init_scale))
                        pyramid_ch = self.total_channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                                    num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, self.total_channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    @staticmethod
    def add_argparse_args(parser):
        # parser.add_argument("--no-centered", dest="centered", action="store_false", help="The data is not centered")
        return parser

    def forward(self, x, time_cond=None):
        """
        - x: b,2*D,F,T: contains x and y OR x: b,D,F,T contains only x
        """
        # timestep/noise_level embedding; only for continuous training

        modules = self.all_modules
        m_idx = 0

        # Convert real and imaginary parts into channel dimensions
        x_chans = []
        for chan in range(self.spatial_channels):
            x_chans.append(torch.cat([ 
                torch.cat([x[:,[chan+in_chan],:,:].real, x[:,[chan+in_chan],:,:].imag ], dim=1) for in_chan in range(self.input_channels // 2)],
                    dim=1)
                )
        x = torch.cat(x_chans, dim=1) #4*D

        if self.time_conditional and time_cond is not None:

            if self.embedding_type == 'fourier':
                # Gaussian Fourier features embeddings.
                used_sigmas = time_cond
                #temb = modules[m_idx](torch.log(used_sigmas))
                temb = modules[m_idx](used_sigmas)
                m_idx += 1
            elif self.embedding_type == 'positional':
                # Sinusoidal positional embeddings.
                timesteps = time_cond
                used_sigmas = self.sigmas[time_cond.long()]
                temb = layers.get_timestep_embedding(timesteps, self.nf)
            else:
                raise ValueError(f'embedding type {self.embedding_type} unknown.')

            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        hs = [modules[m_idx](x)]  # Input layer: Conv2d
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                #print(hs[-1].shape, temb.shape)
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                # edit: check H dim (-2) not W dim (-1)
                if h.shape[-2] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:  # Downsampling
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == 'input_skip':   # Combine h with x
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1
        h = modules[m_idx](h)  # Attention block 
        m_idx += 1
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            # edit: from -1 to -2
            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)

        # Convert to complex number
        h = self.output_layer(h) #b,D=1,C_out,T
        h = torch.reshape(h, (h.size(0), 2, self.spatial_channels, h.size(2), h.size(3)))
        h = torch.permute(h, (0, 2, 3, 4, 1)).contiguous() # b,2,D,F,T -> b,D,F,T,2
        h = torch.view_as_complex(h) #b,D,F,T
        return h





class NCSNppTime(NCSNpp):
    """Same as NCSNpp, but wrapped with a STFT/ISTFT layers"""

    def __init__(self, stft=None, **kwargs):
        assert stft is not None, "stft must be provided"
        super().__init__( 
        **kwargs)
        self.stft_kwargs = stft

        self.window = get_window("hann", self.stft_kwargs.n_fft)
        #self.stft_kwargs = {
        #    "n_fft": self.n_fft,
        #    "hop_length": kwargs["hop_length"],
        #    "center":True
        #    }



    def stft(self, sig):
        window = self.window.to(sig.device)
        C=sig.shape[1]
        sig=einops.rearrange(sig, "b c t -> (b c) t")   
        spec= torch.stft(sig, **{**self.stft_kwargs, "window": window}, return_complex=True)
        # spec= torch.stft(sig, **{**vars(self.stft_kwargs), "window": window}, return_complex=True)
        spec=einops.rearrange(spec, "(b c) f t -> b c f t", c=C)
        #pad in the time axis if the resulting spec is not a multiple of 16
        N_pad = 16 
        if spec.shape[-1] % N_pad != 0:
            num_pad= N_pad - spec.shape[-1] % N_pad
            spec= torch.nn.functional.pad(spec, (0, num_pad, 0, 0), mode="constant", value=0)
        spec = spec.type(torch.complex64)
        return spec


    def istft(self, spec, length=None):
        window = self.window.to(spec.device)
        c=spec.shape[1]
        spec=einops.rearrange(spec, "b c f t -> (b c) f t")
        sig= torch.istft(spec, **{**self.stft_kwargs, "window": window}, length=length)
        # sig= torch.istft(spec, **{**vars(self.stft_kwargs), "window": window}, length=length)
        sig=einops.rearrange(sig, "(b c) t -> b c t", c=c)
        return sig[..., :length]

    def forward(self, x, time_cond=None):

        B,C,T=x.shape

        x_spec=self.stft(x)
        x_spec=super().forward(x_spec, time_cond=time_cond)
        x_time=self.istft(x_spec, length=T)

        return x_time

        

if __name__ == "__main__":

    import argparse
    import torchaudio

    stft_kwargs = {
        "n_fft": 126,
        "hop_length": 32,
        "center": True
    }
    image_size = 64
    ch_mult = [1, 2, 2, 2]
    nf = 128

    # stft_kwargs = {
    #     "n_fft": 510,
    #     "hop_length": 128,
    #     "center": True
    # }
    # image_size = 256
    # ch_mult = [1, 2, 2, 2]
    # nf = 128

    x, sr = torchaudio.load("/data/lemercier/databases/vctk_derev_with_rir/audio/tt/clean/p376_295_3074_t60=1.06.wav")
    # x, sr = torchaudio.load("/data/lemercier/databases/RIRs_organized/ACE_organized/data/Crucif_403a_1_RIR.wav")
    x = x[0].unsqueeze(0)
    x = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(x)

    window = get_window("hann", stft_kwargs["n_fft"])
    x_stft = torch.stft(x, window=window, return_complex=True, **stft_kwargs)
    xhat = torch.istft(x_stft, window=window, **stft_kwargs)

    torchaudio.save("x_original.wav", x, 16000)
    torchaudio.save("x_reconstructed.wav", xhat, 16000)

    print(torch.nn.MSELoss()(x[..., : xhat.size(-1)], xhat))

    stft = argparse.Namespace(**stft_kwargs)

    dnn = NCSNppTime(stft, input_channels=2)

    T = 2.1
    x = torch.randn(2, 1, int(16000 * T))
    sigma = torch.randn(2,)

    xhat = dnn.cuda()(x.cuda(), sigma.cuda())
    print(int(16000 * T), x.shape, xhat.shape)