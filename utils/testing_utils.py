import torch
import numpy as np
import torchaudio
from glob import glob
import os

def add_measurement_noise(y, measurement_noise_type: str, measurement_noise_snr: float, measurement_noise_path, fs: int):
    if measurement_noise_type == "none":
        return y
    
    elif measurement_noise_type == "gaussian":
        unit_std_noise_signal = torch.randn_like(y)

    else:
        noise_files = sorted(glob(os.path.join(measurement_noise_path, "*.wav")))
        noise_path = noise_files[np.random.randint(len(noise_files))]
        noise, fs_N = torchaudio.load(noise_path)
        if fs_N != fs:
            noise = torchaudio.functional.resample(noise, fs_N, fs)
                    
        # Crop or pad
        if noise.size(-1) < y.size(-1):
            noise = torch.nn.functional.pad(noise, ((0, y.size(-1) - noise.size(-1))))
        else:
            offset = np.random.randint(noise.size(-1) - y.size(-1))
            noise = noise[..., offset: offset + y.size(-1)]

        # standardize
        unit_std_noise_signal = noise / noise.std()

    noise_std = y.std() * 10**(-measurement_noise_snr / 20)
    y += noise_std * unit_std_noise_signal

    return y



def add_rir_error(RIR_original, rir_error_type: str, rir_error_snr: float, T_separation: float, fs: int):

    RIR = RIR_original.clone()

    if rir_error_type == "all":
        unit_std_noise_signal = torch.randn_like(RIR)
        noise_std = RIR.std() * 10**(-rir_error_snr / 20)
        RIR += noise_std * unit_std_noise_signal

    elif rir_error_type == "early":

        RIR_early = RIR[..., : int(T_separation*fs)]

        noise_decay = 1e-3 * 10 #10ms decay for noise
        noise_signal_steady = torch.randn_like(RIR_early)[..., : int((T_separation - noise_decay)*fs)]
        noise_decaying_tail = torch.linspace(1., 0., int(noise_decay*fs)).to(RIR.device) * torch.randn_like(RIR_early)[..., : int(noise_decay*fs)]
        noise_signal = torch.cat([ noise_signal_steady, noise_decaying_tail], dim=-1)
        
        noise_std = RIR_early.std() * 10**(-rir_error_snr / 20)

        RIR = torch.cat([ RIR_early + noise_std * noise_signal, RIR[..., int(T_separation*fs): ] ], dim=-1)

    elif rir_error_type == "late":
        RIR_late = RIR[..., int(T_separation*fs): ]
        
        noise_decay = 1e-3 * 10 #10ms decay for noise
        noise_signal_steady = torch.randn_like(RIR_late)[..., int(noise_decay*fs): ]
        noise_increasing_tail = torch.linspace(0., 1., int(noise_decay*fs)).to(RIR.device) * torch.randn_like(RIR_late)[..., : int(noise_decay*fs)]
        noise_signal = torch.cat([ noise_increasing_tail, noise_signal_steady ], dim=-1)
        
        noise_std = RIR_late.std() * 10**(-rir_error_snr / 20)

        RIR = torch.cat([ RIR[..., : int(T_separation*fs)], RIR_late + noise_std * noise_signal ], dim=-1)

    return RIR


    
stft_kwargs = {"n_fft": 510, "hop_length": 128, "window": torch.hann_window(510), "return_complex": True}

def lsd(s_hat, s, eps=1e-10):
    S_hat, S = torch.stft(torch.from_numpy(s_hat), **stft_kwargs), torch.stft(torch.from_numpy(s), **stft_kwargs)
    logPowerS_hat, logPowerS = 2*torch.log(eps + torch.abs(S_hat)), 2*torch.log(eps + torch.abs(S))
    return torch.mean( torch.sqrt(torch.mean(torch.abs( logPowerS_hat - logPowerS ))) ).item()

def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s)/np.linalg.norm(s)**2   
    sdr = 10*np.log10(np.linalg.norm(alpha*s)**2/np.linalg.norm(
        alpha*s - s_hat)**2)
    return sdr

def mean_std(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

def wer(r, h):
    '''
    by zszyellow
    https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return float(d[len(r)][len(h)]) / len(r)

def replace_denormals(x: torch.tensor, threshold=1e-8):
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y
