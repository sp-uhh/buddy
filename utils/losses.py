import torch

def get_frequency_weighting(freqs, freq_weighting=None):
    if freq_weighting is None:
        return torch.ones_like(freqs).to(freqs.device)
    elif freq_weighting=="sqrt":
        return torch.sqrt(freqs)
    elif freq_weighting=="exp":
        freqs = torch.exp(freqs)
        return freqs - freqs[:, 0, :].unsqueeze(-2) 
    elif freq_weighting=="log":
        return torch.log(1+freqs)
    elif freq_weighting=="linear":
        return freqs


def get_loss(loss_args, operator=None):

        if loss_args.name == "none":
            return None
        
        if hasattr(loss_args, "loss_1"): #We have a hybrid of multiple losses
            return lambda x, x_hat: torch.sum( torch.stack([ get_loss( getattr(loss_args, key), operator=operator)(x, x_hat) for key in list(loss_args.keys()) ])  )

        else:
            if "stft" in loss_args.name:
                def loss_fn(x, x_hat, freq_weighting=None):
                    X = operator.apply_stft(x)
                    X_hat = operator.apply_stft(x_hat)
                    freqs = torch.linspace(0, 1, X.shape[-2]).to(X.device).unsqueeze(-1).unsqueeze(0).expand(X.shape)+1
                    freqs = get_frequency_weighting(freqs, freq_weighting=loss_args.get("freq_weighting", None))

                    X = X * freqs
                    X_hat = X_hat * freqs

                    if loss_args.name == "l2_stft_sum":
                        loss= torch.sum((X-X_hat).abs()**2)
                    
                    elif loss_args.name == "l2_stft_mag_sum":
                        loss= torch.sum((X.abs()-X_hat.abs())**2)

                    elif loss_args.name == "l2_stft_logmag_sum":
                        loss= torch.sum((torch.log10(X.abs()+1e-8)-torch.log10(X_hat.abs()+1e-8))**2)

                    elif loss_args.name == "l2_comp_stft_sum":
                        compression_factor = loss_args.get("compression_factor", None)
                        assert compression_factor is not None and compression_factor > 0. and compression_factor <= 1., f"Compression factor weird: {compression_factor}"
                        X_comp =  (X.abs()+1e-8)**compression_factor * torch.exp(1j*X.angle())
                        X_hat_comp= (X_hat.abs()+1e-8)**compression_factor * torch.exp(1j*X_hat.angle())
                        loss= torch.sum((X_comp - X_hat_comp).abs()**2)

                    elif loss_args.name == "l2_comp_stft_mean":
                        compression_factor = loss_args.get("compression_factor", None)
                        assert compression_factor is not None and compression_factor > 0. and compression_factor <= 1., f"Compression factor weird: {compression_factor}"
                        X_comp =  (X.abs()+1e-8)**compression_factor * torch.exp(1j*X.angle())
                        X_hat_comp= (X_hat.abs()+1e-8)**compression_factor * torch.exp(1j*X_hat.angle())
                        loss= torch.mean((X_comp - X_hat_comp).abs()**2)

                    elif loss_args.name == "l2_comp_stft_summean":
                        compression_factor = loss_args.get("compression_factor", None)
                        assert compression_factor is not None and compression_factor > 0. and compression_factor <= 1., f"Compression factor weird: {compression_factor}"
                        X_comp =  (X.abs()+1e-8)**compression_factor * torch.exp(1j*X.angle())
                        X_hat_comp= (X_hat.abs()+1e-8)**compression_factor * torch.exp(1j*X_hat.angle())
                        loss= torch.mean(torch.sum((X_comp - X_hat_comp).abs()**2, dim=-2))

                    elif loss_args.name == "l2_log_stft_sum":
                        X_comp =  torch.log(1+X.abs())* torch.exp(1j*X.angle())
                        X_hat_comp= torch.log(1+X_hat.abs())* torch.exp(1j*X_hat.angle())
                        loss= torch.sum((X_comp - X_hat_comp).abs()**2)

                    else:
                        raise NotImplementedError(f"rec_loss {loss_args.name} not implemented")

                    weight=loss_args.get("weight", 1.)

                    return weight*loss
                return lambda x, x_hat: loss_fn(x, x_hat)

            else:
                if loss_args.name == "l2_sum":
                    def loss_fn(x, x_hat):
                        loss= torch.sum((x-x_hat)**2)
                        weight=loss_args.get("weight", 1.)
                        return weight*loss
                    
                elif loss_args.name == "l2_mean":
                    def loss_fn(x, x_hat):
                        loss= torch.mean((x-x_hat)**2)
                        weight=loss_args.get("weight", 1.)
                        return weight*loss
                        
                else:
                    raise NotImplementedError(f"rec_loss {loss_args.name} not implemented")

                return lambda x, x_hat: loss_fn(x, x_hat)
