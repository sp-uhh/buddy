import torch 
#from src.nsgt.cq  import NSGT

from .fscale  import LogScale , FlexLogOctScale

from .nsgfwin import nsgfwin
from .nsdual import nsdual
from .util import calcwinrange

import math
from math import ceil

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))



class CQT_nsgt():
    def __init__(self,numocts, binsoct,  mode="critical", window="hann", flex_Q=None, fs=44100, audio_len=44100, device="cpu", dtype=torch.float32):
        """
            args:
                numocts (int) number of octaves
                binsoct (int) number of bins per octave. Can be a list if mode="flex_oct"
                mode (string) defines the mode of operation:
                     "critical": (default) critical sampling (no redundancy) returns a list of tensors, each with different time resolution (slow implementation)
                     "critical_fast": notimplemented
                     "matrix": returns a 2d-matrix maximum redundancy (discards DC and Nyquist)
                     "matrix_pow2": returns a 2d-matrix maximum redundancy (discards DC and Nyquist) (time-resolution is rounded up to a power of 2)
                     "matrix_complete": returns a 2d-matrix maximum redundancy (with DC and Nyquist)
                     "matrix_slow": returns a 2d-matrix maximum redundancy (slow implementation)
                     "oct": octave-wise rasterization ( modearate redundancy) returns a list of tensors, each from a different octave with different time resolution (discards DC and Nyquist)
                     "oct_complete": octave-wise rasterization ( modearate redundancy) returns a list of tensors, each from a different octave with different time resolution (with DC and Nyquist)
                fs (float) sampling frequency
                audio_len (int) sample length
                device
        """

        fmax=fs/2 -10**-6 #the maximum frequency is Nyquist
        self.Ls=audio_len #the length is given

        fmin=fmax/(2**numocts)
        fbins=int(binsoct*numocts) 
        self.numocts=numocts
        self.binsoct=binsoct
       
        if mode=="flex_oct":
            self.scale = FlexLogOctScale(fs, self.numocts, self.binsoct, time_reductions)
        else:
            self.scale = LogScale(fmin, fmax, fbins)

        self.fs=fs

        self.device=torch.device(device)
        self.mode=mode
        self.dtype=dtype

        self.frqs,self.q = self.scale() 

        self.g,rfbas,self.M = nsgfwin(self.frqs, self.q, self.fs, self.Ls, dtype=self.dtype, device=self.device, min_win=4, window=window)

        sl = slice(0,len(self.g)//2+1)

        # coefficients per slice
        self.ncoefs = max(int(math.ceil(float(len(gii))/mii))*mii for mii,gii in zip(self.M[sl],self.g[sl]))        
        if mode=="matrix" or mode=="matrix_complete" or mode=="matrix_slow":
            #just use the maximum resolution everywhere
            self.M[:] = self.M.max()
        elif mode=="matrix_pow2":
            self.size_per_oct=[]
            self.M[:]=next_power_of_2(self.M.max())

        elif mode=="oct" or mode=="oct_complete":
            #round uo all the lengths of an octave to the next power of 2
            self.size_per_oct=[]
            idx=1
            for i in range(numocts):
                value=next_power_of_2(self.M[idx:idx+binsoct].max())

                #value=M[idx:idx+binsoct].max()
                self.size_per_oct.append(value)
                self.M[idx:idx+binsoct]=value
                self.M[-idx-binsoct:-idx]=value
                idx+=binsoct


        # calculate shifts
        self.wins,self.nn = calcwinrange(self.g, rfbas, self.Ls, device=self.device)
        # calculate dual windows
        self.gd = nsdual(self.g, self.wins, self.nn, self.M, dtype=self.dtype, device=self.device)

        #filter DC
        self.Hlpf=torch.zeros(self.Ls, dtype=self.dtype, device=self.device)
        self.Hlpf[0:len(self.g[0])//2]=self.g[0][:len(self.g[0])//2]*self.gd[0][:len(self.g[0])//2]*self.M[0]
        self.Hlpf[-len(self.g[0])//2:]=self.g[0][len(self.g[0])//2:]*self.gd[0][len(self.g[0])//2:]*self.M[0]
        #filter nyquist
        nyquist_idx=len(self.g)//2
        Lg=len(self.g[nyquist_idx])
        self.Hlpf[self.wins[nyquist_idx][0:(Lg+1)//2]]+=self.g[nyquist_idx][(Lg)//2:]*self.gd[nyquist_idx][(Lg)//2:]*self.M[nyquist_idx]
        self.Hlpf[self.wins[nyquist_idx][-(Lg-1)//2:]]+=self.g[nyquist_idx][:(Lg)//2]*self.gd[nyquist_idx][:(Lg)//2]*self.M[nyquist_idx]

        self.Hhpf=1-self.Hlpf

        #FORWARD!! this is from nsgtf
        #self.forward = lambda s: nsgtf(s, self.g, self.wins, self.nn, self.M, mode=self.mode , device=self.device)
        #sl = slice(0,len(self.g)//2+1)
        if mode=="matrix" or mode=="oct" or mode=="matrix_pow2":
            sl = slice(1,len(self.g)//2) #getting rid of the DC component and the Nyquist
        else:
            sl = slice(0,len(self.g)//2+1)

        self.maxLg_enc = max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(self.M[sl], self.g[sl]))
    
        self.loopparams_enc = []
        for mii,gii,win_range in zip(self.M[sl],self.g[sl],self.wins[sl]):
            Lg = len(gii)
            col = int(ceil(float(Lg)/mii))
            assert col*mii >= Lg
            assert col == 1
            p = (mii,win_range,Lg,col)
            self.loopparams_enc.append(p)
    

        def get_ragged_giis(g, wins, ms, mode):
            #ragged_giis = [torch.nn.functional.pad(torch.unsqueeze(gii, dim=0), (0, self.maxLg_enc-gii.shape[0])) for gii in gd[sl]]
            #ragged_giis=[]
            c=torch.zeros((len(g),self.Ls//2+1),dtype=self.dtype,device=self.device)
            ix=[]
            if mode=="oct":
                for i in range(self.numocts):
                    ix.append(torch.zeros((self.binsoct,self.size_per_oct[i]),dtype=torch.int64,device=self.device))
            elif mode=="matrix" or mode=="matrix_pow2":
                ix.append(torch.zeros((len(g),self.maxLg_enc),dtype=torch.int64,device=self.device))

            elif mode=="oct_complete" or mode=="matrix_complete":
                ix.append(torch.zeros((1,ms[0]),dtype=torch.int64,device=self.device))
                count=0
                for i in range(1,len(g)-1):
                    if count==0 or ms[i] == ms[i-1]:
                        count+=1
                    else:
                        ix.append(torch.zeros((count,ms[i-1]),dtype=torch.int64,device=self.device))
                        count=1

                ix.append(torch.zeros((count,ms[i-1]),dtype=torch.int64,device=self.device))

                ix.append(torch.zeros((1,ms[-1]),dtype=torch.int64,device=self.device))

            j=0
            k=0
            for i,(gii, win_range) in enumerate(zip(g,wins)):
                if i>0:
                    if ms[i]!=ms[i-1] or ((mode=="oct_complete" or mode=="matrix_complete") and (j==0 or i==len(g)-1)):
                        j+=1
                        k=0

                gii=torch.fft.fftshift(gii).unsqueeze(0)
                Lg=gii.shape[1]

                if (i==0 or i==len(g)-1) and (mode=="oct_complete" or mode=="matrix_complete"):
                    #special case for the DC and Nyquist, as we don't want to use the mirrored frequencies, take this into account during forward! we would just need to conjugate or sth!
                    if i==0:
                        c[i,win_range[Lg//2:]]=gii[...,Lg//2:]

                        ix[j][0,:(Lg+1)//2]=win_range[Lg//2:].unsqueeze(0)
                        ix[j][0,-(Lg//2):]=torch.flip(win_range[Lg//2:].unsqueeze(0),(-1,))
                    if i==len(g)-1:
                        c[i,win_range[:(Lg+1)//2]]=gii[...,:(Lg+1)//2]

                        ix[j][0,:(Lg+1)//2]=torch.flip(win_range[:(Lg+1)//2].unsqueeze(0),(-1,)) #rethink this
                        ix[j][0,-(Lg//2):]=win_range[:(Lg)//2].unsqueeze(0)
                else:
                    c[i,win_range]=gii 

                    ix[j][k,:(Lg+1)//2]=win_range[Lg//2:].unsqueeze(0)
                    ix[j][k,-(Lg//2):]=win_range[:Lg//2].unsqueeze(0)

                k+=1
                #a=torch.unsqueeze(gii, dim=0)
                #b=torch.nn.functional.pad(a, (0, self.maxLg_enc-gii.shape[0]))
                #ragged_giis.append(b)
            #dirty unsqueeze
            return  torch.conj(c), ix


        if self.mode=="matrix" or self.mode=="matrix_complete" or self.mode=="matrix_pow2":
            self.giis, self.idx_enc=get_ragged_giis(self.g[sl], self.wins[sl], self.M[sl],self.mode)
            #self.idx_enc=self.idx_enc[0]
            #self.idx_enc=self.idx_enc.unsqueeze(0).unsqueeze(0)
        elif self.mode=="oct" or self.mode=="oct_complete":
            self.giis, self.idx_enc=get_ragged_giis(self.g[sl], self.wins[sl], self.M[sl], self.mode)
            #self.idx_enc=self.idx_enc.unsqueeze(0).unsqueeze(0)
        elif self.mode=="critical" or self.mode=="matrix_slow":
            #self.giis, self.idx_enc=get_ragged_giis(self.g[sl], self.wins[sl], self.M[sl], self.mode)
            
            ragged_giis = [torch.nn.functional.pad(torch.unsqueeze(gii, dim=0), (0, self.maxLg_enc-gii.shape[0])) for gii in self.g[sl]]
            self.giis = torch.conj(torch.cat(ragged_giis))
            #ragged_giis = [torch.nn.functional.pad(torch.unsqueeze(gii, dim=0), (0, self.maxLg_enc-gii.shape[0])) for gii in self.g[sl]]

            #self.giis = torch.conj(torch.cat(ragged_giis))

        #FORWARD!! this is from nsigtf
        #self.backward = lambda c: nsigtf(c, self.gd, self.wins, self.nn, self.Ls, mode=self.mode,  device=self.device)

        self.maxLg_dec = max(len(gdii) for gdii in self.gd)
        if self.mode=="matrix_pow2":
            self.maxLg_dec=self.maxLg_enc
        #self.maxLg_dec=self.maxLg_enc
        #print(self.maxLg_enc, self.maxLg_dec)
       
        #ragged_gdiis = [torch.nn.functional.pad(torch.unsqueeze(gdii, dim=0), (0, self.maxLg_dec-gdii.shape[0])) for gdii in self.gd]
        #self.gdiis = torch.conj(torch.cat(ragged_gdiis))

        def get_ragged_gdiis(gd, wins, mode, ms=None):
            ragged_gdiis=[]
            ix=torch.zeros((len(gd),self.Ls//2+1),dtype=torch.int64,device=self.device)+self.maxLg_dec//2#I initialize the index with the center to make sure that it points to a 0
            for i,(g, win_range) in enumerate(zip(gd, wins)):
                Lg=g.shape[0]
                gl=g[:(Lg+1)//2]
                gr=g[(Lg+1)//2:]
                zeros = torch.zeros(self.maxLg_dec-Lg ,dtype=g.dtype, device=g.device)  # pre-allocation
                paddedg=torch.cat((gl, zeros, gr),0).unsqueeze(0)
                ragged_gdiis.append(paddedg)

                wr1 = win_range[:(Lg)//2]
                wr2 = win_range[-((Lg+1)//2):]
                if mode=="matrix_complete" and i==0:
                    #ix[i,wr1]=torch.Tensor([self.maxLg_dec-(Lg//2)+i for i in range(len(wr1))]).to(torch.int64) #the end part
                    ix[i,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(torch.int64).to(self.device) #the start part
                elif mode=="matrix_complete" and i==len(gd)-1:
                    ix[i,wr1]=torch.Tensor([self.maxLg_dec-(Lg//2)+i for i in range(len(wr1))]).to(torch.int64).to(self.device) #the end part
                    #ix[i,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(torch.int64) #the start part
                else:
                    ix[i,wr1]=torch.Tensor([self.maxLg_dec-(Lg//2)+i for i in range(len(wr1))]).to(torch.int64).to(self.device) #the end part
                    ix[i,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(torch.int64).to(self.device) #the start part

                
            return torch.conj(torch.cat(ragged_gdiis)).to(self.dtype)*self.maxLg_dec, ix

        def get_ragged_gdiis_critical(gd, ms):
            seq_gdiis=[]
            ragged_gdiis=[]
            mprev=-1
            for i,(g,m) in enumerate(zip(gd, ms)):
                if i>0 and m!=mprev:
                    gdii=torch.conj(torch.cat(ragged_gdiis))
                    if len(gdii.shape)==1:
                        gdii=gdii.unsqueeze(0)
                    #seq_gdiis.append(gdii[0:gdii.shape[0]//2 +1])
                    seq_gdiis.append(gdii)
                    ragged_gdiis=[]
                    
                Lg=g.shape[0]
                gl=g[:(Lg+1)//2]
                gr=g[(Lg+1)//2:]
                zeros = torch.zeros(m-Lg ,dtype=g.dtype, device=g.device)  # pre-allocation
                paddedg=torch.cat((gl, zeros, gr),0).unsqueeze(0)*m
                ragged_gdiis.append(paddedg)
                mprev=m
            
            gdii=torch.conj(torch.cat(ragged_gdiis))
            seq_gdiis.append(gdii)
            #seq_gdiis.append(gdii[0:gdii.shape[0]//2 +1])
            return seq_gdiis

        def get_ragged_gdiis_oct(gd, ms, wins, mode):
            seq_gdiis=[]
            ragged_gdiis=[]
            mprev=-1
            ix=[] 
            if mode=="oct_complete":
                ix+=[torch.zeros((1,self.Ls//2+1),dtype=torch.int64,device=self.device)+ms[0]//2]

            ix+=[torch.zeros((self.binsoct,self.Ls//2+1),dtype=torch.int64,device=self.device)+self.size_per_oct[j]//2 for j in range(len(self.size_per_oct))]
            if mode=="oct_complete":
                ix+=[torch.zeros((1,self.Ls//2+1),dtype=torch.int64,device=self.device)+ms[-1]//2]
            
            #I nitialize the index with the center to make sure that it points to a 0
            j=0
            k=0
            for i,(g,m, win_range) in enumerate(zip(gd, ms, wins)):
                if i>0 and m!=mprev or (mode=="oct_complete" and i==len(gd)-1):
                    #take care when size of DC is the same as the next octave, or last octave has the same size as nyquist!
                    gdii=torch.conj(torch.cat(ragged_gdiis))
                    if len(gdii.shape)==1:
                        gdii=gdii.unsqueeze(0)
                    #seq_gdiis.append(gdii[0:gdii.shape[0]//2 +1])
                    seq_gdiis.append(gdii.to(self.dtype))
                    ragged_gdiis=[]
                    j+=1
                    k=0
                    
                Lg=g.shape[0]
                gl=g[:(Lg+1)//2]
                gr=g[(Lg+1)//2:]
                zeros = torch.zeros(int(m-Lg ),dtype=g.dtype, device=g.device)  # pre-allocation
                paddedg=torch.cat((gl, zeros, gr),0).unsqueeze(0)*m
                ragged_gdiis.append(paddedg)
                mprev=m

                wr1 = win_range[:(Lg)//2]
                wr2 = win_range[-((Lg+1)//2):]
                if mode=="oct_complete" and i==0:
                    #ix[i,wr1]=torch.Tensor([self.maxLg_dec-(Lg//2)+i for i in range(len(wr1))]).to(torch.int64) #the end part
                    #ix[i,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(torch.int64) #the start part
                    ix[0][k,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(self.device).to(torch.int64) #the start part
                elif mode=="oct_complete" and i==len(gd)-1:
                    #ix[i,wr1]=torch.Tensor([self.maxLg_dec-(Lg//2)+i for i in range(len(wr1))]).to(torch.int64) #the end part
                    ix[-1][k,wr1]=torch.Tensor([m-(Lg//2)+i for i in range(len(wr1))]).to(self.device).to(torch.int64) #the end part
                    #ix[i,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(torch.int64) #the start part
                else:
                    #ix[i,wr1]=torch.Tensor([self.maxLg_dec-(Lg//2)+i for i in range(len(wr1))]).to(torch.int64) #the end part
                    #ix[i,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(torch.int64) #the start part

                    ix[j][k,wr1]=torch.Tensor([m-(Lg//2)+i for i in range(len(wr1))]).to(self.device).to(torch.int64) #the end part
                    ix[j][k,wr2]=torch.Tensor([i for i in range(len(wr2))]).to(self.device).to(torch.int64) #the start part
                k+=1
            
            gdii=torch.conj(torch.cat(ragged_gdiis))
            seq_gdiis.append(gdii.to(self.dtype))
            #seq_gdiis.append(gdii[0:gdii.shape[0]//2 +1])

            return seq_gdiis, ix

        if self.mode=="matrix" or self.mode=="matrix_complete":
            self.gdiis, self.idx_dec= get_ragged_gdiis(self.gd[sl], self.wins[sl], self.mode)
            #self.gdiis = self.gdiis[sl]
            #self.gdiis = self.gdiis[0:(self.gdiis.shape[0]//2 +1)]
        elif self.mode=="matrix_pow2":
            self.gdiis, self.idx_dec= get_ragged_gdiis(self.gd[sl], self.wins[sl], self.mode, ms=self.M[sl])
        elif self.mode=="oct" or self.mode=="oct_complete":
            self.gdiis, self.idx_dec=get_ragged_gdiis_oct(self.gd[sl], self.M[sl], self.wins[sl], self.mode)
            for gdiis in self.gdiis:
                gdiis.to(self.dtype)

        elif self.mode=="critical":
            self.gdiis =get_ragged_gdiis_critical(self.gd[sl], self.M[sl])
        elif self.mode=="matrix_slow":
            ragged_gdiis = [torch.nn.functional.pad(torch.unsqueeze(gdii, dim=0), (0, self.maxLg_dec-gdii.shape[0])) for gdii in self.gd]
            self.gdiis = torch.conj(torch.cat(ragged_gdiis))

        self.loopparams_dec = []
        for gdii,win_range in zip(self.gd[sl], self.wins[sl]):
            Lg = len(gdii)
            wr1 = win_range[:(Lg)//2]
            wr2 = win_range[-((Lg+1)//2):]
            p = (wr1,wr2,Lg)
            self.loopparams_dec.append(p)

    def apply_hpf_DC(self, x):
        Lin=x.shape[-1]
        if Lin<self.Ls:
            #pad zeros
            x=torch.nn.functional.pad(x, (0, self.Ls-Lin))
        elif Lin> self.Ls:
            raise ValueError("Input signal is longer than the maximum length. I could have patched it, but I didn't. sorry :(")

        X=torch.fft.fft(x)
        X=X*torch.conj(self.Hhpf)
        out= torch.fft.ifft(X).real
        if Lin<self.Ls:
            out=out[..., :Lin]
        return out


    def apply_lpf_DC(self, x):
        Lin=x.shape[-1]
        if Lin<self.Ls:
            #pad zeros
            x=torch.nn.functional.pad(x, (0, self.Ls-Lin))
        elif Lin> self.Ls:
            raise ValueError("Input signal is longer than the maximum length. I could have patched it, but I didn't. sorry :(")
        X=torch.fft.fft(x)
        X=X*torch.conj(self.Hlpf)
        out= torch.fft.ifft(X).real
        if Lin<self.Ls:
            out=out[..., :Lin]
        return out


    def nsgtf(self,f):
        """
            forward transform
            args:
                t: Tensor shape(B, C, T) time-domain waveform
            returns:
                if mode = "matrix" 
                    ret: Tensor shape (B, C, F, T') 2d spectrogram spectrogram matrix
                else 
                    ret: list([Tensor]) list of tensors of shape (B, C, Fbibs, T') , representing the bands with the same time-resolution.
                    if mode="oct", the elements on the lists correspond to different octaves
                
        """
        

        ft = torch.fft.fft(f)
    
        Ls = f.shape[-1]

        assert self.nn == Ls
    
        if self.mode=="matrix" or self.mode=="matrix_pow2":
            ft=ft[...,:self.Ls//2+1]
            #c = torch.zeros(*f.shape[:2], len(self.loopparams_enc), self.maxLg_enc, dtype=ft.dtype, device=torch.device(self.device))
    
            t=ft.unsqueeze(-2)*self.giis #this has a lot of rendundant operations and, probably, consumes a lot of memory. Anyways, it is parallelizable, so it is not a big deal, I guess.
            #c=torch.gather(t, 3, self.idx_enc)
            a=torch.gather(t, 3, self.idx_enc[0].unsqueeze(0).unsqueeze(0).expand(t.shape[0],t.shape[1],-1,-1)) #To make torch.gather broadcast, I need to add a dimension to the index. 

            #a=torch.cat(a,torch.conj(torch.fliplr(a[...,0:-1])),dim=-1)

            return torch.fft.ifft(a)
    
        elif self.mode=="oct": 
            ft=ft[...,:self.Ls//2 +1]
            #block_ptr = -1
            #bucketed_tensors = []
            ret = []
            #ret2 = []
        
            t=ft.unsqueeze(-2)*self.giis #this has a lot of rendundant operations and, probably, consumes a lot of memory. Anyways, it is parallelizable, so it is not a big deal, I guess.

            for i in range(self.numocts):
                #c=torch.gather(t[...,i*self.binsoct:(i+1)*self.binsoct,:], 3, self.idx_enc[i]) 
                #ret.append(torch.fft.ifft(torch.cat(bucketed_tensors,2)))
                a=torch.gather(t[...,i*self.binsoct:(i+1)*self.binsoct,:], 3, self.idx_enc[i].unsqueeze(0).unsqueeze(0).expand(t.shape[0],t.shape[1],-1,-1)) #To make torch.gather broadcast, I need to add a dimension to the index.
                ret.append(torch.fft.ifft(a))

            return ret
        elif self.mode=="oct_complete": 
            ft=ft[...,:self.Ls//2 +1]
            #block_ptr = -1
            #bucketed_tensors = []
            ret = []
            #ret2 = []
        
            t=ft.unsqueeze(-2)*self.giis #this has a lot of rendundant operations and, probably, consumes a lot of memory. Anyways, it is parallelizable, so it is not a big deal, I guess.

            L=self.idx_enc[0].shape[-1]
            a=torch.gather(t[...,0,:].unsqueeze(-2), 3, self.idx_enc[0].unsqueeze(0).unsqueeze(0).expand(t.shape[0],t.shape[1],-1,-1)) #To make torch.gather broadcast, I need to add a dimension to the index.
            a[...,(L+1)//2:]=torch.conj(a[...,(L+1)//2:])
            #conjugate one of the partsk
            ret.append(torch.fft.ifft(a))

            for i in range(self.numocts):
                #c=torch.gather(t[...,i*self.binsoct:(i+1)*self.binsoct,:], 3, self.idx_enc[i]) 
                #ret.append(torch.fft.ifft(torch.cat(bucketed_tensors,2)))
                a=torch.gather(t[...,i*self.binsoct+1:(i+1)*self.binsoct+1,:], 3, self.idx_enc[i+1].unsqueeze(0).unsqueeze(0).expand(t.shape[0],t.shape[1],-1,-1)) #To make torch.gather broadcast, I need to add a dimension to the index.
                ret.append(torch.fft.ifft(a))
            
            L=self.idx_enc[-1].shape[-1]
            a=torch.gather(t[...,-1,:].unsqueeze(-2), 3, self.idx_enc[-1].unsqueeze(0).unsqueeze(0).expand(t.shape[0],t.shape[1],-1,-1)) #To make torch.gather broadcast, I need to add a dimension to the index. 
            a[...,:(L)//2]=torch.conj(a[...,:(L)//2]) #conjugate one of the parts (here the first)
            ret.append(torch.fft.ifft(a))

            return ret

        elif self.mode=="matrix_complete":
            ft=ft[...,:self.Ls//2+1]
            #c = torch.zeros(*f.shape[:2], len(self.loopparams_enc), self.maxLg_enc, dtype=ft.dtype, device=torch.device(self.device))

    
            t=ft.unsqueeze(-2)*self.giis #this has a lot of rendundant operations and, probably, consumes a lot of memory. Anyways, it is parallelizable, so it is not a big deal, I guess.
            #c=torch.gather(t, 3, self.idx_enc)
            ret=[]
            i=0 #DC be careful!
            L=self.idx_enc[0].shape[-1]
            a=torch.gather(t[...,0,:].unsqueeze(-2), 3, self.idx_enc[0].unsqueeze(0).unsqueeze(0).expand(t.shape[0],t.shape[1],-1,-1)) #To make torch.gather broadcast, I need to add a dimension to the index.
            a[...,(L+1)//2:]=torch.conj(a[...,(L+1)//2:])
            #conjugate one of the partsk
            ret.append(torch.fft.ifft(a))

            #normal
            a=torch.gather(t[...,1:-1,:], 3, self.idx_enc[1].unsqueeze(0).unsqueeze(0).expand(t.shape[0],t.shape[1],-1,-1)) #To make torch.gather broadcast, I need to add a dimension to the index. 
            ret.append(torch.fft.ifft(a))
            #nyquist be careful!
            i=-1 
            a=torch.gather(t[...,-1,:].unsqueeze(-2), 3, self.idx_enc[-1].unsqueeze(0).unsqueeze(0).expand(t.shape[0],t.shape[1],-1,-1)) #To make torch.gather broadcast, I need to add a dimension to the index. 
            a[...,:(L)//2]=torch.conj(a[...,:(L)//2]) #conjugate one of the parts (here the first)
            ret.append(torch.fft.ifft(a))
            #conjugate one of the partsk
            return torch.cat(ret,dim=2)

        elif self.mode=="matrix_slow":
            c = torch.zeros(*f.shape[:2], len(self.loopparams_enc), self.maxLg_enc, dtype=ft.dtype, device=torch.device(self.device))

            for j, (mii,win_range,Lg,col) in enumerate(self.loopparams_enc):
                t = ft[:, :, win_range]*torch.fft.fftshift(self.giis[j, :Lg])

                sl1 = slice(None,(Lg+1)//2)
                sl2 = slice(-(Lg//2),None)

                c[:, :, j, sl1] = t[:, :, Lg//2:]  # if mii is odd, this is of length mii-mii//2
                c[:, :, j, sl2] = t[:, :, :Lg//2]  # if mii is odd, this is of length mii//2

            return torch.fft.ifft(c)
        elif self.mode=="critical": 
            block_ptr = -1
            bucketed_tensors = []
            ret = []
        
            for j, (mii,win_range,Lg,col) in enumerate(self.loopparams_enc):

                c = torch.zeros(*f.shape[:2], 1, mii, dtype=ft.dtype, device=torch.device(self.device))
        
                t = ft[:, :, win_range]*torch.fft.fftshift(self.giis[j, :Lg]) #this needs to be parallelized!
        
                sl1 = slice(None,(Lg+1)//2)
                sl2 = slice(-(Lg//2),None)
        
                c[:, :, 0, sl1] = t[:, :, Lg//2:]  # if mii is odd, this is of length mii-mii//2
                c[:, :, 0, sl2] = t[:, :, :Lg//2]  # if mii is odd, this is of length mii//2
        
                # start a new block
                if block_ptr == -1 or bucketed_tensors[block_ptr][0].shape[-1] != mii:
                    bucketed_tensors.append(c)
                    block_ptr += 1
                else:
                    # concat block to previous contiguous frequency block with same time resolution
                    bucketed_tensors[block_ptr] = torch.cat([bucketed_tensors[block_ptr], c], dim=2)
        
            # bucket-wise ifft
            for bucketed_tensor in bucketed_tensors:
                ret.append(torch.fft.ifft(bucketed_tensor))
        
            return ret

    def nsigtf(self,cseq):
        """
        mode: "matrix"
            args
                cseq: Time-frequency Tensor with shape (B, C, Freq, Time)
            returns
                sig: Time-domain Tensor with shape (B, C, Time)
                
        """


        if self.mode!="matrix" and self.mode!="matrix_slow" and self.mode!="matrix_complete" and self.mode!="matrix_pow2":
            #print(cseq)
            assert type(cseq) == list
            nfreqs = 0
            for i, cseq_tsor in enumerate(cseq):
                cseq_dtype = cseq_tsor.dtype
                cseq[i] = torch.fft.fft(cseq_tsor)
                nfreqs += cseq_tsor.shape[2]
            cseq_shape = (*cseq_tsor.shape[:2], nfreqs)
        else:
            assert type(cseq) == torch.Tensor
            cseq_shape = cseq.shape[:3]
            cseq_dtype = cseq.dtype
            fc = torch.fft.fft(cseq)
        
        fbins = cseq_shape[2]
        #temp0 = torch.empty(*cseq_shape[:2], self.maxLg_dec, dtype=fr.dtype, device=torch.device(self.device))  # pre-allocation
        
        

        # The overlap-add procedure including multiplication with the synthesis windows
        #tart=time.time()
        if self.mode=="matrix_slow":
            fr = torch.zeros(*cseq_shape[:2], self.nn, dtype=cseq_dtype, device=torch.device(self.device))  # Allocate output
            temp0 = torch.empty(*cseq_shape[:2], self.maxLg_dec, dtype=fr.dtype, device=torch.device(self.device))  # pre-allocation

            for i,(wr1,wr2,Lg) in enumerate(self.loopparams_dec[:fbins]):
                t = fc[:, :, i]

                r = (Lg+1)//2
                l = (Lg//2)

                t1 = temp0[:, :, :r]
                t2 = temp0[:, :, Lg-l:Lg]

                t1[:, :, :] = t[:, :, :r]
                t2[:, :, :] = t[:, :, self.maxLg_dec-l:self.maxLg_dec]

                temp0[:, :, :Lg] *= self.gdiis[i, :Lg] 
                temp0[:, :, :Lg] *= self.maxLg_dec

                fr[:, :, wr1] += t2
                fr[:, :, wr2] += t1

        elif self.mode=="matrix" or self.mode=="matrix_complete" or self.mode=="matrix_pow2":
            fr = torch.zeros(*cseq_shape[:2], self.nn//2+1, dtype=cseq_dtype, device=torch.device(self.device))  # Allocate output
            temp0=fc*self.gdiis.unsqueeze(0).unsqueeze(0)
            fr=torch.gather(temp0, 3, self.idx_dec.unsqueeze(0).unsqueeze(0).expand(temp0.shape[0], temp0.shape[1], -1, -1)).sum(2)

        elif self.mode=="oct" or self.mode=="oct_complete":
            fr = torch.zeros(*cseq_shape[:2], self.nn//2+1, dtype=cseq_dtype, device=torch.device(self.device))  # Allocate output
            # frequencies are bucketed by same time resolution
            fbin_ptr = 0
            for j, (fc, gdii_j) in enumerate(zip(cseq, self.gdiis)):
                Lg_outer = fc.shape[-1]
        
                nb_fbins = fc.shape[2]
                temp0 = torch.zeros(*cseq_shape[:2],nb_fbins, Lg_outer, dtype=cseq_dtype, device=torch.device(self.device))  # Allocate output
        
                temp0=fc*gdii_j.unsqueeze(0).unsqueeze(0)
                fr+=torch.gather(temp0, 3, self.idx_dec[j].unsqueeze(0).unsqueeze(0).expand(temp0.shape[0], temp0.shape[1], -1, -1)).sum(2)

        else:
            # speed uniefficient but save mode
            # frequencies are bucketed by same time resolution
            fr = torch.zeros(*cseq_shape[:2], self.nn, dtype=cseq_dtype, device=torch.device(self.device))  # Allocate output
            fbin_ptr = 0
            for j, (fc, gdii_j) in enumerate(zip(cseq, self.gdiis)):
                Lg_outer = fc.shape[-1]
        
                nb_fbins = fc.shape[2]
                temp0 = torch.zeros(*cseq_shape[:2],nb_fbins, Lg_outer, dtype=cseq_dtype, device=torch.device(self.device))  # Allocate output
        
                temp0=fc*gdii_j.unsqueeze(0).unsqueeze(0)

                for i,(wr1,wr2,Lg) in enumerate(self.loopparams_dec[fbin_ptr:fbin_ptr+nb_fbins][:fbins]):
                    r = (Lg+1)//2
                    l = (Lg//2)
        
                    fr[:, :, wr1] += temp0[:,:,i,Lg_outer-l:Lg_outer]
                    fr[:, :, wr2] += temp0[:,:,i, :r]

                fbin_ptr += nb_fbins
        
        #end=time.time()
        #rint("in for loop",end-start)
        ftr = fr[:, :, :self.nn//2+1]
        sig = torch.fft.irfft(ftr, n=self.nn)
        sig = sig[:, :, :self.Ls] # Truncate the signal to original length (if given)
        return sig

    def fwd(self,x):
        """
            x: [B,C,T]
        """
        c = self.nsgtf(x)
        return c

    def bwd(self,c):
        s = self.nsigtf(c) #messing out with the channels agains...
        return s

