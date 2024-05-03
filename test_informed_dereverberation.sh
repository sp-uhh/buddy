
export TORCH_USE_RTLD_GLOBAL=YES
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

ckpt=<pretrained-vctk-checkpoint.pt>
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