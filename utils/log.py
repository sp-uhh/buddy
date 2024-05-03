import os
import torch 
import numpy as np
import plotly.express as px
import soundfile as sf
import pandas as pd
import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt

"""
Logging related functions
"""

def do_stft(noisy, clean=None, win_size=2048, hop_size=512, device="cpu", DC=True):
    window=torch.hamming_window(window_length=win_size)
    window=window.to(noisy.device)
    noisy=torch.cat((noisy, torch.zeros(noisy.shape[0],win_size).to(noisy.device)),1)
    stft_signal_noisy=torch.stft(noisy, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
    stft_signal_noisy=stft_signal_noisy.permute(0,3,2,1)
    
    if clean!=None:

        clean=torch.cat((clean, torch.zeros(clean.shape[0],win_size).to(device)),1)
        stft_signal_clean=torch.stft(clean, win_size, hop_length=hop_size,window=window, center=False,return_complex=False)
        stft_signal_clean=stft_signal_clean.permute(0,3,2,1)

        if DC:
            return stft_signal_noisy, stft_signal_clean
        else:
            return stft_signal_noisy[...,1:], stft_signal_clean[...,1:]
    else:

        if DC:
            return stft_signal_noisy
        else:
            return stft_signal_noisy[...,1:]
 
def error_line(error_y_mode='band', **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {'bar','band','bars','bands',None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar','bars',None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band','bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = False,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig

def plot_loss_by_sigma(sigma_means, sigma_stds, sigma_bins):
    df=pd.DataFrame.from_dict(
                {"sigma": sigma_bins, "loss": sigma_means, "std": sigma_stds
                }
                )

    fig= error_line('bar', data_frame=df, x="sigma", y="loss", error_y="std", log_x=True,  markers=True, range_y=[0, 2])
    
    return fig

def write_audio_file(x, sr, string: str, path='tmp', stereo=False, normalize=False):
    if normalize:
        x=x/torch.max(torch.abs(x))
    if not(os.path.exists(path)): 
        os.makedirs(path)
      
    path=os.path.join(path,string+".wav")
    if stereo:
        '''
        x has shape (B,2,T)
        '''
        x=x.permute(0,2,1) #B,T,2
        x=x.flatten(0,1) #B*T,2
        x=x.cpu().numpy()
    else:
        x=x.flatten()
        x=x.unsqueeze(1)
        x=x.cpu().numpy()
    sf.write(path,x,sr)

    return path

def get_spectrogram_from_raw_audio(x, stft, refr=1):
    X=do_stft(x, win_size=stft.win_size, hop_size=stft.hop_size)
    X=X.permute(0,2,3,1)
    X=X.squeeze(1)
    X=torch.sqrt(X[:,:,:,0]**2 + X[:,:,:,1]**2)
     
    S_db = 10*torch.log10(torch.abs(X)/refr)
    S_db=S_db.permute(0,2,1)
    S_db=torch.flip(S_db, [1])

    for i in range(X.shape[0]): #iterate over batch size, shity way of ploting all the batched spectrograms
        o=S_db[i]
        if i==0:
             res=o
        else:
             res=torch.cat((res,o), 1)
    return res