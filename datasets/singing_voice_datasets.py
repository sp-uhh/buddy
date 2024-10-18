import os
import numpy as np
import torch
import random
import pandas as pd
import glob
import soundfile as sf
import librosa
from tqdm import tqdm


class MultiDataset(torch.utils.data.IterableDataset):
    def __init__(self,
        path_csv,
        priorities,
        normalize=False,
        overfit= False,
        load_len= 552552,
        apply_random_gain= True,
        gain_range= [-3, 3],
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        
        num_files=[]

        metafilelist=[]
        for i, p in enumerate(path_csv):
            filelist_df=pd.read_csv(p)
            #convert the dataframe to list
            metafilelist.append(filelist_df.values.tolist())
            num_files.append(len(filelist_df.values.tolist()))

            assert len(metafilelist[i])>0 , "error in dataloading: empty or nonexistent folder"

        priorities=[priorities[j]*num_files[j] for j in range(len(priorities))]
        priorities=[priorities[j]/sum(priorities) for j in range(len(priorities))]
        priorities=np.array(priorities)
        self.priorities_cumsum=np.cumsum(priorities)

        self.train_samples=metafilelist
       
        self.seg_len=int(load_len)
        self.normalize=normalize
        self.apply_random_gain=apply_random_gain
        self.gain_range=gain_range

    def __iter__(self):
        while True:
            try:
                    randnum=np.random.uniform(0,1-1e-8)
                    #search for the last element in priorities_cumsum that is smaller than randnum
                    #print(self.priorities_cumsum, randnum, np.where(self.priorities_cumsum<randnum))
                    try:
                        idx=int(np.where(self.priorities_cumsum<randnum)[0][-1]+1)
                        #print(idx)
                    except:
                        idx=0
                        #print("except",idx)

                    num=random.randint(0,len(self.train_samples[idx])-1)

                    #for file in self.train_samples:  
                    file=self.train_samples[idx][num]
                    data, samplerate = sf.read(file[0])
                    if samplerate>48000:
                        continue
                    #assert(samplerate==self.fs, "wrong sampling rate")
                    data_clean=data
                    #Stereo to mono
                    if len(data.shape)>1 :
                        data_clean=np.mean(data_clean,axis=1)

                    #print(len(data_clean), self.seg_len)
                    if len(data_clean)<self.seg_len:
                        #circular padding with random offset
                        idx=0
                        segment=np.tile(data_clean,(self.seg_len//data_clean.shape[-1]+1))[...,idx:idx+self.seg_len]
                        #random offset (circular shift)
                        segment=np.roll(segment, np.random.randint(0,self.seg_len), axis=-1)
                    else:
                        idx=np.random.randint(0,len(data_clean)-self.seg_len)
                        segment=data_clean[idx:idx+self.seg_len]
                    segment=segment.astype('float32')
                    if self.normalize:
                        std=segment.std(-1)
                        #check if any in std is 0
                        if std<2e-5:
                            print("std is 0, skipping")
                            continue
                            #raise ValueError("std is 0, skipping")
                        segment/=std

                    if self.apply_random_gain:
                        #apply gain defined in gain_range in dB
                        segment*=10**(np.random.uniform(self.gain_range[0],self.gain_range[1])/20)
        
                    yield  segment, samplerate

            except Exception as e:
                print(e)
                continue

class BenchmarkTest(torch.utils.data.Dataset):
    def __init__(self,
        fs=44100,
        segment_length=262144,
        path="", #path to the dataset
        normalize=False,  #to normalize or not. I don't normalize by default
        ):
        super().__init__()

        self.test_samples=[]
        #iterate over speakers directories
        path_x=os.path.join(path,"clean")
        path_rir=os.path.join(path,"rir")

        self.test_samples=glob.glob(os.path.join(path_x,"*.wav"))
        print("len(test_samples)",len(self.test_samples), path_x)
        ids=[os.path.splitext(os.path.basename(f))[0] for f in self.test_samples]

        self.rir_samples=[os.path.join(path_rir,id+".wav") for id in ids]

        self.segment_length=int(segment_length)
        self.fs=fs

        self.normalize=normalize
        if self.normalize:
            raise NotImplementedError("normalization not implemented yet")

        self.test_audio=[]
        self.test_rir=[]
        self.test_reverb=[]
        self.filenames=[]
        self._fs=[]
        for file_x, file_rir in tqdm(zip(self.test_samples, self.rir_samples)):

            self.filenames.append(os.path.basename(file_x))
            data, samplerate = sf.read(file_x)

            if samplerate != self.fs:
                #"wrong sampling rate"
                data = librosa.resample(data, orig_sr=samplerate, target_sr=self.fs)
            
            file_reverberant=file_x.replace("clean","reverberant")
            data_reverb, samplerate = sf.read(file_reverberant)

            
            assert data_reverb.shape==data.shape, "reverberant and clean files have different shapes"

            assert len(data.shape) == 1, "wrong number of channels"

            if segment_length!=-1:
                if len(data)<self.segment_length:
                    #pad with zeros
                    data=np.pad(data, (0,self.segment_length-len(data)))
                    data_reverb=np.pad(data_reverb, (0,self.segment_length-len(data_reverb)))
                elif len(data)>self.segment_length:
                    #crop
                    data=data[:self.segment_length]
                    data_reverb=data_reverb[:self.segment_length]
    
            rir_data, samplerate = sf.read(file_rir)
            if samplerate != self.fs:
                #"wrong sampling rate"
                rir_data = librosa.resample(rir_data, orig_sr=samplerate, target_sr=self.fs)
    
            assert len(rir_data.shape) == 1, "wrong number of channels"


            self.test_audio.append(data) #use only 50s
            self.test_reverb.append(data_reverb) #use only 50s
            self.test_rir.append(rir_data) #use only 50s

    def __getitem__(self, idx):
        return self.test_audio[idx],self.test_reverb[idx], self.test_rir[idx], self.filenames[idx]
        #return self.test_audio[idx], self.test_rir[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)


class AudioFolderDatasetTest(torch.utils.data.Dataset):
    def __init__(self,
        path,
        stereo= False,
        num_examples= 16,
        batch_size=1,
        fs=44100,
        seg_len=131072,
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        filelist=glob.glob(os.path.join(path,"*.wav"))
        assert len(filelist)>0 , "error in dataloading: empty or nonexistent folder"
        self.train_samples=filelist
        self.seg_len=int(seg_len)
        self.fs=fs

        self.test_samples=[]
        self.filenames=[]
        self._fs=[]
        num_samples=min(num_examples,len(self.train_samples))
        for i in range(num_samples):
            file=self.train_samples[i]
            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            data=data.T
            self._fs.append(samplerate)
            print("test",data.shape)
            if data.shape[-1]>=self.seg_len:
                print("cropping")
                idx=np.random.randint(0,data.shape[-1]-self.seg_len)
                segment=data[...,idx:idx+self.seg_len]
            else:
                print("tiling")
                idx=0
                segment=np.tile(data,(self.seg_len//data.shape[-1]+1))[...,idx:idx+self.seg_len]

            if not stereo and len(data.shape)>1 :
                print("shape",segment.shape)
                segment=np.mean(segment,axis=0)

            if stereo:
                raise NotImplementedError("stereo not implemented yet")

            #std=segment.std(-1)
            #check if any in std is 0
            self.test_samples.append(segment) #use only 50s

    def __getitem__(self, idx):
        #return self.test_samples[idx]
        return self.test_samples[idx], self._fs[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)