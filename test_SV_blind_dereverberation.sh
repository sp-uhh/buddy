
export TORCH_USE_RTLD_GLOBAL=YES
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

#ckpt=<pretrained-vctk-checkpoint.pt>
ckpt=/scratch/work/molinee2/projects/audiodps_2/audiodps_eloi_post_iwaenc/experiments/singing_voice/44k_8s-325000.pt

tester=blind_dereverberation_BUDDy_SV
conf=conf_singing_voice.yaml
name=buddy_wpe-init_noise-prior_N-201_rir-aligned_1exp

PATH_EXPERIMENT=experiments/$name
mkdir $PATH_EXPERIMENT
HYDRA_FULL_ERROR=1 python test.py --config-name=$conf \
            tester=$tester \
            tester.checkpoint=$ckpt \
            tester.sampling_params.T=101 \
            model_dir=$PATH_EXPERIMENT \
            +gpu=0 \
            dset=SV_test-benchmark \
