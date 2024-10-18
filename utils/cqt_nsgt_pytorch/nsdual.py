# -*- coding: utf-8

"""
Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)

--
Original matlab code comments follow:

NSDUAL.M - Nicki Holighaus 02.02.11

Computes (for the painless case) the dual frame corresponding to a given 
non-stationary Gabor frame specified by the windows 'g' and time shifts
'shift'.

Note, the time shifts corresponding to the dual window sequence is the
same as the original shift sequence and as such already given.

This routine's output can be used to achieve reconstruction of a signal 
from its non-stationary Gabor coefficients using the inverse 
non-stationary Gabor transform 'nsigt'.

More information on Non-stationary Gabor transforms
can be found at:

http://www.univie.ac.at/nonstatgab/

minor edit by Gino Velasco 23.02.11
"""

import numpy as np
import torch

#from .util import chkM


def nsdual(g, wins, nn, M=None, dtype=torch.float32, device="cpu"):
    #M = chkM(M,g)

    # Construct the diagonal of the frame operator matrix explicitly
    x = torch.zeros((nn,), dtype=dtype, device=torch.device(device))
    for gi,mii,sl in zip(g, M, wins):
        xa = torch.square(torch.fft.fftshift(gi))
        xa *= mii

        x[sl] += xa

    gd = [gi/torch.fft.ifftshift(x[wi]) for gi,wi in zip(g,wins)]
    return gd
