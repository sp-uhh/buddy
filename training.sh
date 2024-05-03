
export TORCH_USE_RTLD_GLOBAL=YES
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

python train.py --config-name=conf1.yaml \
            dset.train.path=$data_dir/VCTK-Corpus/wav16 \
            dset.test.path=$data_dir/VCTK-Corpus/wav16 \
            exp.batch_size=8 \
            logging.wandb.entity=jean-marie/lemercier \
            model_dir=experiments/edm_vctk_time