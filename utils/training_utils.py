
import torch
import numpy as np
from torch.profiler import tensorboard_trace_handler

def load_state_dict( state_dict, network=None, ema=None, optimizer=None, log=True):
        '''
        utility for loading state dicts for different models. This function sequentially tries different strategies
        args:
            state_dict: the state dict to load
        returns:
            True if the state dict was loaded, False otherwise
        Assuming the operations are don in_place, this function will not create a copy of the network and optimizer (I hope)
        '''
        #print(state_dict)
        if log: print("Loading state dict")
        if log:
            print(state_dict.keys())
        #if there
        try:
            if log: print("Attempt 1: trying with strict=True")
            if network is not None:
                network.load_state_dict(state_dict['network'])
            if optimizer is not None:
                optimizer.load_state_dict(state_dict['optimizer'])
            if ema is not None:
                ema.load_state_dict(state_dict['ema'])
            return True
        except Exception as e:
            if log:
                print("Could not load state dict")
                print(e)
        try:
            if log: print("Attempt 2: trying with strict=False")
            if network is not None:
                network.load_state_dict(state_dict['network'], strict=False)
            #we cannot load the optimizer in this setting
            #self.optimizer.load_state_dict(state_dict['optimizer'], strict=False)
            if ema is not None:
                ema.load_state_dict(state_dict['ema'], strict=False)
            return True
        except Exception as e:
            if log:
                print("Could not load state dict")
                print(e)
                print("training from scratch")
        try:
            if log: print("Attempt 3: trying with strict=False,but making sure that the shapes are fine")
            if ema is not None:
                ema_state_dict = ema.state_dict()
            if network is not None:
                network_state_dict = network.state_dict()
            i=0 
            if network is not None:
                for name, param in state_dict['network'].items():
                    if log: print("checking",name) 
                    if name in network_state_dict.keys():
                        if network_state_dict[name].shape==param.shape:
                                network_state_dict[name]=param
                                if log:
                                    print("assigning",name)
                                i+=1
            network.load_state_dict(network_state_dict)
            if ema is not None:
                for name, param in state_dict['ema'].items():
                        if log: print("checking",name) 
                        if name in ema_state_dict.keys():
                            if ema_state_dict[name].shape==param.shape:
                                ema_state_dict[name]=param
                                if log:
                                    print("assigning",name)
                                i+=1
     
            ema.load_state_dict(ema_state_dict)
     
            if i==0:
                if log: print("WARNING, no parameters were loaded")
                raise Exception("No parameters were loaded")
            elif i>0:
                if log: print("loaded", i, "parameters")
                return True

        except Exception as e:
            print(e)
            print("the second strict=False failed")


        try:
            if log: print("Attempt 4: Assuming the naming is different, with the network and ema called 'state_dict'")
            if network is not None:
                network.load_state_dict(state_dict['state_dict'])
            if ema is not None:
                ema.load_state_dict(state_dict['state_dict'])
        except Exception as e:
            if log:
                print("Could not load state dict")
                print(e)
                print("training from scratch")
                print("It failed 3 times!! but not giving up")
            #print the names of the parameters in self.network

        try:
            if log: print("Attempt 5: trying to load with different names, now model='model' and ema='ema_weights'")
            if ema is not None:
                dic_ema = {}
                for (key, tensor) in zip(state_dict['model'].keys(), state_dict['ema_weights']):
                    dic_ema[key] = tensor
                    ema.load_state_dict(dic_ema)
                return True
        except Exception as e:
            if log:
                print(e)

        try:
            if log: print("Attempt 6: If there is something wrong with the name of the ema parameters, we can try to load them using the names of the parameters in the model")
            if ema is not None:
                dic_ema = {}
                i=0
                for (key, tensor) in zip(state_dict['model'].keys(), state_dict['model'].values()):
                    if tensor.requires_grad:
                        dic_ema[key]=state_dict['ema_weights'][i]
                        i=i+1
                    else:
                        dic_ema[key]=tensor     
                ema.load_state_dict(dic_ema)
                return True
        except Exception as e:
            if log:
                print(e)


        try:
            #assign the parameters in state_dict to self.network using a for loop
            print("Attempt 7: Trying to load the parameters one by one. This is for the dance diffusion model, looking for parameters starting with 'diffusion.' or 'diffusion_ema.'")
            if ema is not None:
                ema_state_dict = ema.state_dict()
            if network is not None:
                network_state_dict = ema.state_dict()
            i=0 
            if network is not None:
                for name, param in state_dict['state_dict'].items():
                    print("checking",name) 
                    if name.startswith("diffusion."):
                        i+=1
                        name=name.replace("diffusion.","")
                        if network_state_dict[name].shape==param.shape:
                            #print(param.shape, network.state_dict()[name].shape)
                            network_state_dict[name]=param
                            #print("assigning",name)
           
                network.load_state_dict(network_state_dict, strict=False)
           
            if ema is not None:
                for name, param in state_dict['state_dict'].items():
                    if name.startswith("diffusion_ema."): 
                        i+=1
                        name=name.replace("diffusion_ema.","")
                        if ema_state_dict[name].shape==param.shape:
                            if log:
                                    print(param.shape, ema.state_dict()[name].shape)
                            ema_state_dict[name]=param
           
                ema.load_state_dict(ema_state_dict, strict=False)
           
            if i==0:
                print("WARNING, no parameters were loaded")
                raise Exception("No parameters were loaded")
            elif i>0:
                print("loaded", i, "parameters")
                return True
        except Exception as e:
            if log:
                print(e)
        if network is not None:
            network.load_state_dict(state_dict, strict=True)
        if ema is not None:
            ema.load_state_dict(state_dict, strict=True)
        return True

def profile(args_logging):

    profile=False
    if args_logging.profiling.enabled:
        try:
            print("Profiling is being enabled")
            wait=args_logging.profiling.wait
            warmup=args_logging.profiling.warmup
            active=args_logging.profiling.active
            repeat=args_logging.profiling.repeat

            schedule =  torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat)
            profiler = torch.profiler.profile(
            schedule=schedule, on_trace_ready=tensorboard_trace_handler("wandb/latest-run/tbprofile"), profile_memory=True, with_stack=False)
            profile=True
            profile_total_steps = (wait + warmup + active) * (1 + repeat)
        except Exception as e:

            print("Could not setup profiler")
            print(e)
            profiler=None
            profile=False
            profile_total_steps=0

    return profiler, profile, profile_total_steps