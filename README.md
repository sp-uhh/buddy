# audiodps

I started using python 3.10, I hope that doesn't break your code.


I tried to prepare a simple version of my codebase. The train script works. The test, not yet. 
I will push some changes tomorrow, and I'll do some tests with a model trained on VCTK, just to make sure that the things I share are working.

Briefly, the code is structured like this (I'll expand on the details later):
* 'diff_params' : contains all diffusion schedule parameters and utilities that are required for training, and for computing neural function evaluations. This includes a method for computing the loss ('loss_fn'), and a method for computing the tweedie estimate ('denoiser')
* 'dset' : the data loader
* 'network': the backbone DNN. I added for now the ncsnpp. 
* 'training/trainer': runs the training loop
* 'testing/tester': runs testing experiments (bandwidth extension, dereverberation)
* 'testing/sampler': I will soon include DPS based on a second-order Heun sampler. I will also prepare RED-diff. 

# Jm

* Hi Eloi thanks for all this: I can write Euler-Heun-DPS if you want, so that you can focus on RED-diff if you already have experience with that. (
* On my end, I also retrained the models with similar code and everything works so it's a good basis for understanding I believe,
* I will check diff_params, maybe I could integrate Song's VE there as well, just for reference (in my recent experience I notice it is much harder to train that Karras' EDM and yields worse results)  

# Eloi

* I also have Euler-Heun-DPS ready in my other messy codebase, so it would be easy for me to add it as well. But feel free to write it too, if it does not require much extra work, so we can have both versions.
* I would just write a new diff_params file for Song's VE, it may not be too hard to integrate. The tricky thing may be to make the samplers work with different kinds of trained models.
* Let me know if this code looks fine to you, and feel free to modify anything
* Another detail, I'm now training a model using diffusion on the waveform, but just using the STFT and ISTFT as preprocessing in the backbone neural network. This differs from the way you worked, were you defined the diffusion process in the complex spectrogram domain. I experimented with both approaches while working with your repo, and they seemed to perform similar. But it may be nice to include spectrogram diffusion here as well.
* Also, should we use some other method for communication? Slack for example?

# Jm

* Cool, yes I will not touch Euler-Heun-DPS for now then. I already created a new diff_params file for Song's VE. I created a shared superclass called SDE to harmonize the basic functions. I will push all of that on my new branch "jm"
* I am currently trying to run a training, and it seems the 'datasets' folder with the corresponding PyTorch scripts is messing (the one called by the dset config files). Maybe you are calling a default torchaudio.datasets.vctk.VCTK as the one existing here? https://vincentqb.github.io/audio/_modules/torchaudio/datasets/vctk.html But even with the current setup it is not automatically fetching that class. Not sure what is wrong there.
* As for the spectrogra,/time discussion, I think we can easily switch from one to the other by creating a 'transform' method in the model or something like that. WIll look into it
* I just invited you to a new Slack! I will invite the others as well. Just need to check how many workspaces I can keep with my free trial.
