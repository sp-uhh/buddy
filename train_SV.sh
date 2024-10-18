
export TORCH_USE_RTLD_GLOBAL=YES
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

python train.py --config-name=conf_singing_voice.yaml \
            logging.wandb.entity=your-wandb-login