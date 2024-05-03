import os
import re
import json
import hydra
import torch

import utils.setup as setup
from training.trainer import Trainer
from testing.tester import Tester

def _main(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

    args.exp.model_dir=args.model_dir

    train_set=hydra.utils.instantiate(args.dset.train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.exp.batch_size, num_workers=args.exp.num_workers, pin_memory=True, worker_init_fn=setup.worker_init_fn,timeout=0, prefetch_factor=20)
    train_loader=iter(train_loader)

    test_set=hydra.utils.instantiate(args.dset.test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1,  num_workers=args.exp.num_workers, pin_memory=True, worker_init_fn=setup.worker_init_fn)

    # Diffusion parameters
    diff_params=hydra.utils.instantiate(args.diff_params) #instantiate in trainer better

    # Network
    if args.network._target_=='networks.unet_octCQT.UNet_octCQT':
        network=hydra.utils.instantiate(args.network, sample_rate=args.exp.sample_rate, audio_len=args.exp.audio_len, device=device ) #instantiate 

    else:
        network=hydra.utils.instantiate(args.network) #instantiate in trainer better

    network=network.to(device)

    # Tester
    args.tester.sampling_params.same_as_training = True #Make sure that we use the same HP for sampling as the ones used in training
    tester=Tester(args, network, diff_params, test_set=test_loader, device=device, in_training=True)

    # Trainer
    trainer=hydra.utils.instantiate(args.exp.trainer, args, train_loader, network, diff_params, tester, device) # This works

    # Print options.
    print()
    print('Training options:')
    print()
    print(f'Output directory:        {args.model_dir}')
    print(f'Network architecture:    {args.network._target_}')
    print(f'Dataset:    {args.dset.train._target_}')
    print(f'Diffusion parameterization:  {args.diff_params._target_}')
    print(f'Batch size:              {args.exp.batch_size}')
    print()

    # Train.
    trainer.training_loop()

@hydra.main(config_path="conf", config_name="conf", version_base=str(hydra.__version__))
def main(args):
    _main(args)

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
