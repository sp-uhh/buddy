# BUDDy: Single-channel Blind Unsupervised Dereverberation with Diffusion Models #

This is the official code for our paper [BUDDy: Single-channel Blind Unsupervised Dereverberation with Diffusion Models](https://arxiv.org/abs/2405.04272)

We invite you to check our [companion website](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/iwaenc2024-buddy.html) for listening samples and insights into the paper

## 1 - Requirements

Install required Python packages with `python -m pip install -r requirements.txt`

## 2 - Checkpoints

You can access our pretrained checkpoint, trained on VCTK anechoic speech, at [the following link](https://drive.google.com/drive/u/2/folders/1fEvzbiIy77A1i5aiOwPf78OKQjCemOmQ)

## 3 - Testing

You can launch blind dereverberation with `bash test_blind_dereverberation.sh`.
You can launch informed dereverberation with `bash test_informed_dereverberation.sh`.
In both cases, do not forget to add the path to the pretrained model checkpoint in the bash file (i.e. replace `ckpt=<pretrained-vctk-checkpoint.pt>` with your path).
The directory tree in `audio_examples/` contains an example test set to reproduce the results.  

## 4 - Training

You can retrain an unsupervised diffusion model on your own dataset with `bash train.sh`.
Do not forget to fill in the path to your training and testing dataset (i.e. replace `dset.train.path=/your/path/to/anechoic/training/set` and so on)

## 5 - Citing

If you used this repo for your own work, do not forget to cite us:

@bibtex
```
@article{moliner2024buddy,
    title={{BUDD}y: Single-channel Blind Unsupervised Dereverberation with Diffusion Models},
    author={Moliner, Eloi and Lemercier, Jean-Marie and Welker, Simon and Gerkmann, Timo and V\"alim\"aki, Vesa},
    year={2024},
    journal={arXiv 2405.04272}
}
```
