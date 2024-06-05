from datetime import date
import re
import torch
import torchaudio
import os
import numpy as np
import wandb
import copy
from glob import glob
from tqdm import tqdm
import omegaconf
import hydra
import soundfile as sf

from testing.operators.subband_filtering import BlindSubbandFiltering
from testing.operators.reverb import RIROperator

import utils.log as utils_logging
import utils.training_utils as tr_utils

class Tester():
    def __init__(
        self, args, network, diff_params, test_set=None, device=None, in_training=False,
    ):
        self.args=args
        self.network=network
        self.diff_params=copy.copy(diff_params)
        self.device=device
        self.test_set = test_set
        self.in_training = in_training

        self.sampler=hydra.utils.instantiate(args.tester.sampler, self.network, self.diff_params, self.args)
    
    def load_latest_checkpoint(self):
        #load the latest checkpoint from self.args.model_dir
        try:
            # find latest checkpoint_id
            save_basename = f"{self.args.exp.exp_name}-*.pt"
            save_name = f"{self.args.model_dir}/{save_basename}"
            list_weights = glob(save_name)
            id_regex = re.compile(f"{self.args.exp.exp_name}-(\d*)\.pt")
            list_ids = [int(id_regex.search(weight_path).groups()[0])
                        for weight_path in list_weights]
            checkpoint_id = max(list_ids)

            state_dict = torch.load(
                f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt", map_location=self.device)
            try:
                self.network.load_state_dict(state_dict['ema'])
            except Exception as e:
                print(e)
                print("Failed to load in strict mode, trying again without strict mode")
                self.network.load_state_dict(state_dict['model'], strict=False)

            print(f"Loaded checkpoint {checkpoint_id}")
            return True
        except (FileNotFoundError, ValueError):
            raise ValueError("No checkpoint found")

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device)
        try:
            self.it=state_dict['it']
        except:
            self.it=0
        print("loading checkpoint")
        return tr_utils.load_state_dict(state_dict, ema=self.network)

    def load_checkpoint_legacy(self, path):
        state_dict = torch.load(path, map_location=self.device)

        try:
            print("load try 1")
            self.network.load_state_dict(state_dict['ema'])
        except:
            #self.network.load_state_dict(state_dict['model'])
            try:
                print("load try 2")
                dic_ema = {}
                for (key, tensor) in zip(state_dict['model'].keys(), state_dict['ema_weights']):
                    dic_ema[key] = tensor
                self.network.load_state_dict(dic_ema)
            except:
                print("load try 3")
                dic_ema = {}
                i=0
                for (key, tensor) in zip(state_dict['model'].keys(), state_dict['model'].values()):
                    if tensor.requires_grad:
                        dic_ema[key]=state_dict['ema_weights'][i]
                        i=i+1
                    else:
                        dic_ema[key]=tensor     
                self.network.load_state_dict(dic_ema)
        try:
            self.it=state_dict['it']
        except:
            self.it=0


    ##############################
    ### UNCONDITIONAL SAMPLING ###
    ##############################

    def sample_unconditional(self, mode):
        audio_len = self.args.exp.audio_len if not "audio_len" in self.args.tester.unconditional.keys() else self.args.tester.unconditional.audio_len
        shape = [self.args.tester.unconditional.num_samples, audio_len]
        preds = self.sampler.predict_unconditional(shape, self.device)

        if not self.in_training:
            for i in range(len(preds)):
                path_generated = utils_logging.write_audio_file(preds[i], self.args.exp.sample_rate, f"unconditional_{i}", path=self.paths["unconditional"])

        return preds





    #######################
    ### DEREVERBERATION ###
    #######################

    def test_dereverberation(self, mode, blind=False):

        if self.test_set is None:
            print("No test set specified")
            return
        if len(self.test_set) == 0:
            print("No samples found in test set")
            return
        
        for i, (original, rir,  filename) in enumerate(tqdm(self.test_set)):

            seg = torch.from_numpy(original).float().to(self.device)
            seg = self.args.tester.posterior_sampling.warm_initialization.scaling_factor * seg / seg.std() #Normalize the input to match sigma_data of dataset

            #read and prepare the RIR
            RIR=torch.Tensor(rir).to(self.device)

            with torch.no_grad():

                # Forward pass with true RIR
                operator_ref = RIROperator(self.args.tester.informed_dereverberation.op_hp, time_kernel_size=RIR.shape[-1], sample_rate=self.args.exp.sample_rate)
                operator_ref.update_params(RIR)
                y = operator_ref.degradation(seg.unsqueeze(0))

                if blind: # Initialize operator
                    assert self.args.tester.blind_dereverberation.operator == "subband_filtering"
                    operator_blind = BlindSubbandFiltering(self.args.tester.informed_dereverberation.op_hp, sample_rate=self.args.exp.sample_rate)
                    with torch.no_grad():
                        operator_blind.update_H(use_noise=True)

            pred = self.sampler.predict_conditional(y, operator_blind if blind else operator_ref, shape=(1,seg.shape[-1]), blind=blind)

            path_original=utils_logging.write_audio_file(seg, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"original"])
            path_degraded=utils_logging.write_audio_file(y, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"degraded"])
            path_reconstructed=utils_logging.write_audio_file(pred, self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"reconstructed"])
            
            utils_logging.write_audio_file(RIR.detach().cpu(), self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"true_rir"])
            if blind:
                utils_logging.write_audio_file(self.sampler.operator.get_time_RIR().detach().cpu(), self.args.exp.sample_rate, os.path.basename(filename)[: -4], path=self.paths[mode+"estimated_rir"])
            
            print(path_reconstructed)



    def prepare_directories(self, mode, unconditional=False, blind=False):
            
            today=date.today() 
            self.paths={}

            if "overriden_name" in self.args.tester.keys() and self.args.tester.overriden_name is not None:
                self.path_sampling = os.path.join(self.args.model_dir, self.args.tester.overriden_name)
            else:
                self.path_sampling = os.path.join(self.args.model_dir,'test'+today.strftime("%d_%m_%Y"))
            if not os.path.exists(self.path_sampling):
                os.makedirs(self.path_sampling)

            self.paths[mode]=os.path.join(self.path_sampling,mode,self.args.exp.exp_name)

            if not os.path.exists(self.paths[mode]):
                os.makedirs(self.paths[mode])

            if not unconditional:
                self.paths[mode+"original"]=os.path.join(self.paths[mode],"original")
                if not os.path.exists(self.paths[mode+"original"]):
                    os.makedirs(self.paths[mode+"original"])
                self.paths[mode+"degraded"]=os.path.join(self.paths[mode],"degraded")
                if not os.path.exists(self.paths[mode+"degraded"]):
                    os.makedirs(self.paths[mode+"degraded"])
                self.paths[mode+"reconstructed"]=os.path.join(self.paths[mode],"reconstructed")
                if not os.path.exists(self.paths[mode+"reconstructed"]):
                    os.makedirs(self.paths[mode+"reconstructed"])
                    
                if "dereverberation" in mode:
                    self.paths[mode+"true_rir"]=os.path.join(self.paths[mode],"true_rir")
                    if not os.path.exists(self.paths[mode+"true_rir"]):
                        os.makedirs(self.paths[mode+"true_rir"])

                    if mode == "blind_dereverberation":
                        self.paths[mode+"estimated_rir"]=os.path.join(self.paths[mode],"estimated_rir")
                        if not os.path.exists(self.paths[mode+"estimated_rir"]):
                            os.makedirs(self.paths[mode+"estimated_rir"])

    def save_experiment_args(self, mode):
        with open(os.path.join(self.paths[mode], ".argv"), 'w') as f: #Keep track of the arguments we used for this experiment
            omegaconf.OmegaConf.save(config=self.args, f=f.name)

    def do_test(self, it=0):

        self.it = it
        for m in self.args.tester.modes:

            if m == "unconditional":
                print("testing unconditional")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=True)
                    self.save_experiment_args(m)
                return self.sample_unconditional(m)
            
            elif m == "informed_dereverberation":
                print("testing informed dereverberation")
                if not self.in_training:
                    self.prepare_directories(m)
                    self.save_experiment_args(m)
                self.test_dereverberation(m)

            elif m == "blind_dereverberation":
                print("testing blind dereverberation")
                if not self.in_training:
                    self.prepare_directories(m)
                    self.save_experiment_args(m)
                self.test_dereverberation(m, blind=True)

            else:
                print("Warning: unknown mode: ", m)
