import torch

def pad_spec(Y, mode="replicate"):
    # padding_target = 64
    padding_target = 16
    T = Y.size(-1)
    if T%padding_target !=0:
        num_pad = padding_target-T%padding_target
    else:
        num_pad = 0
    return torch.nn.functional.pad(Y, (0, num_pad, 0, 0), mode=mode)

def pad_time(Y, mode="replicate"):
    padding_target = 8192
    T = Y.size(-1)
    if T%padding_target !=0:
        num_pad = padding_target-T%padding_target
    else:
        num_pad = 0
    return torch.nn.functional.pad(Y, (0, num_pad, 0, 0), mode=mode)

def replace_denormals(x: torch.tensor, threshold=1e-8):
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y