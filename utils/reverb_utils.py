import torch

def hilbert(h):
    window = 2 * torch.heaviside(torch.linspace(-1, 1, steps=h.size(-1)), values=torch.ones(1)).to(h.device)
    window = torch.flip(window, dims=(-1,))
    windowed_fft = window * torch.fft.fft(h)
    return torch.fft.ifft(windowed_fft)

def minimum_phase_version(h):
    """
    h is the time-domain RIR.
    We ensure that the RIR has minimum-phase-lag, which helps with stability, as its inverse is then causal and stable.
    """
    T_orig = h.size(-1)
    h = torch.nn.functional.pad(h, (0, T_orig))
    H = torch.fft.fft(h)
    log_H_abs = torch.log(torch.abs(H) + 1e-8)
    minimum_phase = - torch.imag( hilbert(log_H_abs) )
    exp_minimum_phase = torch.exp(1j*minimum_phase)

    minimum_phase_h = torch.real(torch.fft.ifft( torch.abs(H).type(exp_minimum_phase.dtype) * exp_minimum_phase )) # |H(w)|*e^(jPhi(w))
    minimum_phase_h = minimum_phase_h[: -T_orig]
    return minimum_phase_h

def fast_apply_RIR(y, filter, rm_delay=False, zero_pad=False):

    if rm_delay:
        filter = filter[ torch.argmax(filter): ]

    filter = filter.unsqueeze(0).unsqueeze(0)
    B = filter.to(y.device)
    y = y.unsqueeze(1)
    
    # Get the size of the input signal and filter
    N = y.size(2)
    M = filter.size(2)
    
    # Compute the size of the FFT
    if zero_pad:
        fft_size=torch.tensor(2*N+2*M-1)
    else:
        fft_size=torch.tensor(N+M-1)
    fft_size=int(2**torch.ceil(torch.log2(fft_size)))
    
    # Perform FFT on the input signal and filter
    Y = torch.fft.fft(y, fft_size, dim=2)
    H = torch.fft.fft(B, fft_size, dim=2)
    
    # Perform element-wise multiplication in the frequency domain
    Y_conv = Y * H
    
    # Perform inverse FFT to get the convolution result
    y_conv = torch.fft.ifft(Y_conv, fft_size, dim=2)
    
    # Take the real part of the result
    y_conv = y_conv[:, :, :N].real
    
    # Squeeze the unnecessary dimensions
    y_conv = y_conv.squeeze(1)

    return y_conv