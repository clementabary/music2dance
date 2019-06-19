import torch
import torch.autograd as autograd


def gradient_penalty(critic, bsize, real, fake, device=None, is_seq=False, lp=False):
    '''
    Gradient penalty for both stick & sequence WGAN frameworks with regularization
    Classic WGAN-GP if lp=False
    WGAN-LP (WGAN-GP with max(0, .)) if lp=True
    '''

    real = real.view(real.size(0), -1)
    fake = fake.view(fake.size(0), -1)
    alpha = torch.rand(bsize, 1)
    if device:
        alpha = alpha.expand(real.size()).to(device)
    else:
        alpha = alpha.expand(real.size())
    interpol = alpha * real.detach() + (1 - alpha) * fake.detach()
    if not is_seq:
        interpol = interpol.view(interpol.size(0), 23, 3)
    else:
        interpol = interpol.view(interpol.size(0), 69, -1)
    interpol.requires_grad_(True)
    interpol_critic = critic(interpol)
    gradients = autograd.grad(outputs=interpol_critic, inputs=interpol,
                              grad_outputs=torch.ones(interpol_critic.size(),
                                                      device=device),
                              create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    if lp:  # WGAN-LP
        bgrad = gradients.norm(2, dim=1) - 1
        bgrad[bgrad < 0] = 0
        return (bgrad ** 2).mean()
    else:  # WGAN-GP
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def smoothed_lips_penalty(critic, real, fake, device=None, lp=False):
    '''
    Recurrent WGAN-GP involves double backward for RNN modules not yet implemented
    in PyTorch : smoothed version here
    '''
    fx_norm = (critic(real) - critic(fake)).abs()
    x_norm = torch.norm((real - fake), dim=(1, 2))
    pseudo_grad = 1 - (fx_norm.squeeze() / x_norm)
    if lp:
        pseudo_grad[pseudo_grad < 0] = 0
    return (pseudo_grad ** 2).mean()


def total_variation_loss(series):
    '''
    Not correct version
    '''
    bsize, seq_length, dim = series.size()
    tv = 0
    for idx in range(seq_length-1):
        tv += torch.abs(series[:, idx+1, :] - series[:, idx, :])
    return tv
