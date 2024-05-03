
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.optim
import math

class AdamWScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free AdamW
    As the name suggests, no scheduler is needed with this optimizer. 
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.
    
    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining 
            parameter groups.
        lr (float): 
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): 
            Term added to the denominator outside of the root operation to 
            improve numerical stability. (default: 1e-8).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average 
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
    """
    def __init__(self,
                 params, 
                 lr=0.0025, 
                 betas=(0.9, 0.999), 
                 eps=1e-8,
                 weight_decay=0,
                 warmup_steps=0,
                 r=0.0,
                 weight_lr_power=2.0,
                 ):

        defaults = dict(lr=lr, 
                        betas=betas, 
                        eps=eps,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode = True,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to x
                        p.data.lerp_(end=state['z'], weight=1-1/beta1)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to y
                        p.data.lerp_(end=state['z'], weight=1-beta1)
                group['train_mode'] = True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            decay = group['weight_decay']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']
            weight_lr_power = group['weight_lr_power']
            
            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0
            
            bias_correction2 = 1 - beta2 ** (k+1)
            lr = group['lr']*sched*math.sqrt(bias_correction2)
            
            lr_max = group['lr_max'] = max(lr, group['lr_max'])
            
            weight = ((k+1)**r) * (lr_max**weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            ckp1 = weight/weight_sum

            if not group['train_mode']:
                raise Exception("Not in train mode!")

            for p in group['params']:
                if p.grad is None:
                    continue

                y = p.data # Notation to match theory
                grad = p.grad.data

                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(y)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                z = state['z']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                denom = exp_avg_sq.sqrt().add_(eps)

                # Reuse grad buffer for memory efficiency
                grad_normalized = grad.div_(denom)

                # Weight decay calculated at y
                if decay != 0:
                    grad_normalized.add_(y, alpha=decay)

                # These operations update y in-place,
                # without computing x explicitly.
                y.lerp_(end=z, weight=ckp1)
                y.add_(grad_normalized, alpha=lr*(beta1*(1-ckp1)-1))

                # z step
                z.sub_(grad_normalized, alpha=lr)

            group['k'] = k+1
        return loss

class pSGLD(torch.optim.Optimizer):
    """Implements pSGLD algorithm based on https://arxiv.org/pdf/1512.07666.pdf
    from: https://github.com/alisiahkoohi/Langevin-dynamics/blob/master/langevin_sampling/precondSGLD.py

    Built on the PyTorch RMSprop implementation
    (https://pytorch.org/docs/stable/_modules/torch/optim/rmsprop.html#RMSprop)
    """

    def __init__(self,
                 params,
                 lr: float = 1e-2,
                 beta: float = 0.99,
                 Lambda: float = 1e-15,
                 weight_decay: float = 0,
                 centered: bool = False):
        """
        Initializes the pSGLD optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float, optional): Learning rate. Default is 1e-2.
            beta (float, optional): Exponential moving average coefficient.
                Default is 0.99.
            Lambda (float, optional): Epsilon value. Default is 1e-15.
            weight_decay (float, optional): Weight decay coefficient. Default
                is 0.
            centered (bool, optional): Whether to use centered gradients.
                Default is False.
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= Lambda:
            raise ValueError("Invalid epsilon value: {}".format(Lambda))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= beta:
            raise ValueError("Invalid beta value: {}".format(beta))

        defaults = dict(lr=lr,
                        beta=beta,
                        Lambda=Lambda,
                        centered=centered,
                        weight_decay=weight_decay)
        super(pSGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(pSGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            float: Value of G (as defined in the algorithm) after the step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'pSGLD does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['V'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                V = state['V']
                beta = group['beta']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                V.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(beta).add_(1 - beta, grad)
                    G = V.addcmul(grad_avg, grad_avg,
                                  value=-1).sqrt_().add_(group['Lambda'])
                else:
                    G = V.sqrt().add_(group['Lambda'])

                p.data.addcdiv_(grad, G, value=-group['lr'])

                noise_std = 2 * group['lr'] / G
                noise_std = noise_std.sqrt()
                noise = p.data.new(p.data.size()).normal_(mean=0,
                                                          std=1) * noise_std
                p.data.add_(noise)

        return G

class GGDO(torch.optim.Optimizer):
    '''
        Implements the Gaussian Gradient Distruption Optimization
        from: https://github.com/Anirudhsekar96/Noisy_SGD/blob/master/ggdo2.py
    '''
    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0,eps=1e-6, noise=0.1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= noise:
            raise ValueError("Invalid noise value: {}".format(noise))
        
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps, noise=noise)
        super(GGDO, self).__init__(params, defaults)
            
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Gaussian Gradients does not support sparse gradients')
                state = self.state[p]

                #if weight_decay != 0:
                #    grad.add_(weight_decay, p.data)

                # State initialization
                if len(state) == 0:
                    # Intialize mean and variance to zero
                    state['mean'] = torch.zeros_like(p.data)
                    state['variance'] = torch.zeros_like(p.data)
                    state['std'] = torch.zeros_like(p.data)
                    state['step'] = 0
                    
                
                mean = state['mean'] # Works now
                var = state['variance']
                std = state['std']
                
                state['step'] += 1
                
                # Getting mean,std at previous step
                old_mean = mean.clone()
                old_std = std.clone()
                
                
                # Calculating gradients
                new_updt = torch.normal(mean=old_mean, std=old_std)
                updt = grad.add(group['noise'],new_updt)
                if weight_decay != 0:
                    updt.add_(weight_decay, p.data)

                # Updating mean
                mean = mean.mul(group['momentum']).add(updt)
                
                part_var1 = grad.add(-old_mean)
                part_var2 = grad.add(-mean)
                
                new_std = torch.pow(old_std,2).mul(group['momentum']).addcmul(1,part_var1,part_var2).add(group['eps'])                
                new_std = torch.pow(torch.abs_(new_std),1/2)
                std.add_(-1,std).add_(new_std)
                
		
                
                p.data.add_(-group['lr'],updt)
                
        
        return loss

from torch.optim import Optimizer

class CustomAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup = 0, termperature = 0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.temperature = termperature
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup = warmup)
        super(CustomAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CustomAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                #if self.temperature > 0:
                #    noise= self.temperature*torch.randn_like(grad)
                #    grad.add_(noise)

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
                
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)
                
                #avg = square_avg.sqrt().add_(group['eps'])
                #sgld_updt = torch.normal(mean=0,std=avg)

                #p_data_fp32.addcdiv_(-step_size, exp_avg, denom).add_(sgld_updt)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)


                #step_size = group['lr'] / (1 - beta1 ** state['step'])

                # Noise term for SGLD
                noise = self.temperature*torch.randn_like(p_data_fp32) * math.sqrt(step_size) * exp_avg_sq.sqrt()

                if self.temperature > 0:
                    p_data_fp32.add_(noise)

                #add noise scaled by the square root of the learning rate and the momentum
                #noise=noise*(1/(group["eps"]+state["exp_avg_sq"].sqrt()))
                #print(noise.std(), "exp_avg",state["exp_avg"].std(), state["exp_avg_sq"].std())


                #p_data_fp32.add_(noise)    



                p.data.copy_(p_data_fp32)

        return loss