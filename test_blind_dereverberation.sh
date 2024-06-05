
export TORCH_USE_RTLD_GLOBAL=YES
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

#ckpt=<pretrained-vctk-checkpoint.pt>
ckpt=/home/molinee2/projects/audiodps/experiments/VCTK_16k/VCTK_16k_4s_time-190000.pt

tester=blind_dereverberation_BUDDy
conf=conf_VCTK.yaml
name=buddy_wpe-init_noise-prior_N-201_rir-aligned_1exp

PATH_EXPERIMENT=experiments/$name
mkdir $PATH_EXPERIMENT
HYDRA_FULL_ERROR=1 python test.py --config-name=$conf \
            tester=$tester \
            tester.checkpoint=$ckpt \
            tester.sampling_params.T=201 \
            model_dir=$PATH_EXPERIMENT \
            +gpu=0 \
            dset=vctk_16k_4s_test-benchmark \
            dset.test.path=audio_examples\
            dset.test.num_examples=2 \
