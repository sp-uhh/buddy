
import torch
import plotly.express as px
import pandas as pd

def apply_filter_BABE(x, H, NFFT):
    X=apply_stft(x, NFFT)
    xrec=apply_filter_istft(X, H, NFFT)
    xrec=xrec[:,:x.shape[-1]]

    return xrec

def apply_stft(x, NFFT):
    #hamming window
    window = torch.hamming_window(window_length=NFFT)
    window=window.to(x.device)

    x=torch.cat((x, torch.zeros(*x.shape[:-1],NFFT).to(x.device)),1) #is padding necessary?
    X = torch.stft(x, NFFT, hop_length=NFFT//2,  window=window,  center=False, onesided=True, return_complex=True)
    X=torch.view_as_real(X)

    return X

def apply_filter_istft(X, H, NFFT):
    #hamming window
    window = torch.hamming_window(window_length=NFFT)
    window=window.to(X.device)

    X=X*H.unsqueeze(-1).unsqueeze(-1).expand(X.shape)
    X=torch.view_as_complex(X)
    x=torch.istft(X, NFFT, hop_length=NFFT//2,  window=window, center=False, return_complex=False)

    return x

def design_filter_BABE(filter_params, f):
    """
    fc: cutoff frequency 
        if fc is a scalar, the filter has one slopw
        if fc is a list of scalars, the filter has multiple slopes
    A: attenuation in dB
        if A is a scalar, the filter has one slopw
        if A is a list of scalars, the filter has multiple slopes
    """
    multiple_slopes=False
    #check if fc and A are lists
    fc=filter_params[0]
    A=filter_params[1]
    if isinstance(fc, list) and isinstance(A, list):
        multiple_slopes=True
    #check if fc is a tensor and A is a tensor
    try:
        if fc.shape[0]>1:
            multiple_slopes=True
    except:
        pass

    if multiple_slopes:
        H=torch.zeros(f.shape).to(f.device)
        H[f<fc[0]]=1
        H[f>=fc[0]]=10**(A[0]*torch.log2(f[f>=fc[0]]/fc[0])/20)
        for i in range(1,len(fc)):
            H[f>=fc[i]]=10**(A[i]*torch.log2(f[f>=fc[i]]/fc[i])/20)*H[f>=fc[i]][0]

    else:
        #if fc and A are scalars
        H=torch.zeros(f.shape).to(f.device)
        H[f<fc]=1
        H[f>=fc]=10**(A*torch.log2(f[f>=fc]/fc)/20)
        
    return H

def design_filter(fc, A, f):
   H=torch.zeros(f.shape).to(f.device)
   H[f<fc]=1
   H[f>=fc]=10**(A*torch.log2(f[f>=fc]/fc)/20)
   return H

def apply_filter_and_norm_STFTmag_fweighted(X,Xref, H, freq_weight="linear"):
    #X: (N,513, T) "clean" example
    #Xref: (N,513, T)  observations
    #H: (513,) filter

    #get the absolute value of the STFT
    X=torch.sqrt(X[...,0]**2+X[...,1]**2)
    Xref=torch.sqrt(Xref[...,0]**2+Xref[...,1]**2)

    X=X*H.unsqueeze(-1).expand(X.shape)
    freqs=torch.linspace(0, 1, X.shape[1]).to(X.device)
    #apply frequency weighting to the cost function
    if freq_weight=="linear":
        X=X*freqs.unsqueeze(-1).expand(X.shape)
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)
    elif freq_weight=="None":
        pass
    elif freq_weight=="log":
        X=X*torch.log2(1+freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.log2(1+freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="sqrt":
        X=X*torch.sqrt(freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.sqrt(freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="log2":
        X=X*torch.log2(freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.log2(freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="log10":
        X=X*torch.log10(freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.log10(freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="cubic":
        X=X*freqs.unsqueeze(-1).expand(X.shape)**3
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)**3
    elif freq_weight=="quadratic":
        X=X*freqs.unsqueeze(-1).expand(X.shape)**2
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)**2
    elif freq_weight=="logcubic":
        X=X*torch.log2(1+freqs.unsqueeze(-1).expand(X.shape)**3)
        Xref=Xref*torch.log2(1+freqs.unsqueeze(-1).expand(Xref.shape)**3)
    elif freq_weight=="logquadratic":
        X=X*torch.log2(1+freqs.unsqueeze(-1).expand(X.shape)**2)
        Xref=Xref*torch.log2(1+freqs.unsqueeze(-1).expand(Xref.shape)**2)
    elif freq_weight=="squared":
        X=X*freqs.unsqueeze(-1).expand(X.shape)**4
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)**4

    norm=torch.linalg.norm(X.reshape(-1)-Xref.reshape(-1),ord=2)
    return norm