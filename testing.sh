
export TORCH_USE_RTLD_GLOBAL=YES
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

# ckpt=/export/home/lemercier/code/audiodps/experiments/edm_vctk_mse_b16/VCTK_16k_4s_time-190000.pt
# tester=blind_dereverberation_buddy
# conf=conf_VCTK.yaml
# name=buddy_wpe-init_noise-prior_N-101_rir-aligned_2exp
# PATH_EXPERIMENT=experiments/$name
# mkdir $PATH_EXPERIMENT
# HYDRA_FULL_ERROR=1 python test.py --config-name=$conf \
#             tester=$tester \
#             tester.checkpoint=$ckpt \
#             model_dir=$PATH_EXPERIMENT \
#             +gpu=0 \
#             dset=vctk_16k_4s_test-benchmark \
#             dset.test._target_=datasets.vctk_jm.VCTKTest \
#             dset.test.path=/data3/databases/VCTK-Reverb-eloi-subset/test/clean \
#             tester.informed_dereverberation.path_RIRs=/data3/databases/VCTK-Reverb-eloi-subset/test/rir \
#             tester.informed_dereverberation.files=[] \
#             dset.test.num_examples=2 \
#             tester.sampling_params.T=101 \
#             tester.informed_dereverberation.op_hp.init_params.T60_breakpoints=[0.1,0.2] \
#             tester.informed_dereverberation.op_hp.init_params.multiexp_weighting=[2,1] \


ckpt=/export/home/lemercier/code/audiodps/experiments/edm_vctk_mse_b16/VCTK_16k_4s_time-190000.pt
tester=informed_dereverberation_DPS
conf=conf_VCTK.yaml
name=inf_warm-init_N-201_rir-aligned
PATH_EXPERIMENT=experiments/$name
mkdir $PATH_EXPERIMENT
HYDRA_FULL_ERROR=1 python test.py --config-name=$conf \
            tester=$tester \
            tester.checkpoint=$ckpt \
            model_dir=$PATH_EXPERIMENT \
            +gpu=0 \
            dset=vctk_16k_4s_test-benchmark \
            dset.test._target_=datasets.vctk.VCTKTest \
            dset.test.path=/data3/databases/VCTK-Reverb-eloi-subset/test/clean \
            tester.informed_dereverberation.path_RIRs=/data3/databases/VCTK-Reverb-eloi-subset/test/rir \
            tester.informed_dereverberation.files=[] \
            dset.test.num_examples=2 