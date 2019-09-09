import torch
import torch.autograd as autograd


def gradient_penalty(critic, bsize, real, fake, audio=None,
                     is_seq=False, is_cond=False, lp=False, device=None):
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
    if audio is not None:
        audio.requires_grad_(True)
    if (audio is not None) and (not is_cond):
        interpol_critic = critic(interpol, audio)
        inputs = (interpol, audio)
    elif (audio is not None) and is_cond:
        interpol_critic, _ = critic(interpol, audio)
        inputs = (interpol, audio)
    elif (audio is None) and (not is_cond):
        interpol_critic = critic(interpol)
        inputs = interpol
    elif (audio is None) and is_cond:
        interpol_critic, _ = critic(interpol)
        inputs = interpol
    gradients = autograd.grad(outputs=interpol_critic, inputs=inputs,
                              grad_outputs=torch.ones(interpol_critic.size(),
                                                      device=device),
                              create_graph=True, retain_graph=True,
                              only_inputs=True)
    if audio is None:
        gradients = gradients[0].view(gradients[0].size(0), -1)
        if lp:  # WGAN-LP
            bgrad = gradients.norm(2, dim=1) - 1
            bgrad[bgrad < 0] = 0
            return (bgrad ** 2).mean()
        else:  # WGAN-GP
            gradients_norm = torch.sqrt(
                torch.sum(gradients ** 2, dim=1) + 1e-12)
            return ((gradients_norm - 1) ** 2).mean()
    else:
        g0 = gradients[0].view(gradients[0].size(0), -1)
        g1 = gradients[1].view(gradients[0].size(0), -1)
        gnorm0 = torch.sqrt(torch.sum(g0 ** 2, dim=1) + 1e-12)
        gnorm1 = torch.sqrt(torch.sum(g1 ** 2, dim=1) + 1e-12)
        return ((gnorm0 - 1) ** 2).mean() + ((gnorm1 - 1) ** 2).mean()


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


def tv_loss(sequence):
    '''
    Total variation regularizer
    sequence is supposed to be of shape (bsize, coordinates, seq_length)
    '''
    diffs = (sequence[:, :, 1:] - sequence[:, :, :-1]).abs()
    return diffs.mean()


def jerkiness(sequence):
    diffs = (sequence[:, :, 3:] - 3 * sequence[:, :, 2:-1] +
             3 * sequence[:, :, 1:-2] - sequence[:, :, :-3])
    diffs = diffs**2
    return diffs.sum(dim=1).mean()
