import os
import numpy as np
import torch
import random
import pandas as pd
import soundfile as sf
import librosa

class MaestroDataset(torch.utils.data.IterableDataset):
    def __init__(self,
        segment_length=131072,
        fs=22050,
        path="/data/maestro-v3.0.0",
        years=[2004,2005,2006,2007,2008,2009,2010,2011,2012],
        split="train",
        normalize=False,
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        metadata_file=os.path.join(path,"maestro-v3.0.0.csv")
        metadata=pd.read_csv(metadata_file)

        metadata=metadata[metadata["year"].isin(years)]
        metadata=metadata[metadata["split"]==split]
        filelist=metadata["audio_filename"]

        filelist=filelist.map(lambda x:  os.path.join(path,x)     , na_action='ignore')


        self.train_samples=filelist.to_list()
       
        self.seg_len=int(segment_length)
        self.fs=fs


    def __iter__(self):
        if self.overfit:
           data_clean=self.overfit_sample
        while True:
            if not self.overfit:
                num=random.randint(0,len(self.train_samples)-1)
                #for file in self.train_samples:  
                file=self.train_samples[num]
                data, samplerate = sf.read(file)
                #print(file,samplerate)
                data_clean=data
                #Stereo to mono
                if len(data.shape)>1 :
                    data_clean=np.mean(data_clean,axis=1)
                #resample if necessary
                if samplerate!=self.fs:
                    #it is very slow to resample here. We should do it in GPU. I leave it here for now for simplicity
                    data_clean=librosa.resample(data_clean,samplerate,self.fs)
    

            num_frames=np.floor(len(data_clean)/self.seg_len) 
            
            if num_frames>4:
                for i in range(8):
                    #get 8 random batches to be a bit faster
                    idx=np.random.randint(0,len(data_clean)-self.seg_len)
                    segment=data_clean[idx:idx+self.seg_len]
                    segment=segment.astype('float32')
                    yield  segment, samplerate
            else:
                pass


class MaestroDatasetTest(torch.utils.data.Dataset):
    def __init__(self,
        segment_length=131072,
        fs=22050,
        path="/data/maestro-v3.0.0",
        years=[2004,2005,2006,2007,2008,2009,2010,2011,2012],
        split="test",
        normalize=False,
        num_examples=4,
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.seg_len=int(segment_length)

        metadata_file=os.path.join(path,"maestro-v3.0.0.csv")
        metadata=pd.read_csv(metadata_file)

        metadata=metadata[metadata["year"].isin(years)]
        metadata=metadata[metadata["split"]==split]
        filelist=metadata["audio_filename"]

        filelist=filelist.map(lambda x:  os.path.join(path,x)     , na_action='ignore')


        self.fs=fs
        self.filelist=filelist.to_list()

        self.test_samples=[]
        self.filenames=[]
        self.f_s=[]
        for i in range(num_examples):
            file=self.filelist[i]
            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            if len(data.shape)>1 :
                data=np.mean(data,axis=1)
            data=data[10*samplerate:10*samplerate+int(1.1*self.seg_len*samplerate/self.fs)] #use only 50s
            if samplerate!=self.fs:
                data=librosa.resample(data,samplerate,self.fs)
            

            assert data.shape[0]>=self.seg_len, "segment length is larger than the audio, something is wrong in the dataset, please check"
            data=data[:self.seg_len]
            print("data shape",data.shape)
            #picking up a segment at 10s
            self.test_samples.append(data) #use only 50s
            self.f_s.append(samplerate)
       

    def __getitem__(self, idx):
        return self.test_samples[idx],self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)
