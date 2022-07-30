import torch
import numpy as np
import copy

# Thanks to https://gitlab.com/qbeer/loss-landscape/-/blob/main/loss_landscape/landscape_utils.py

def init_directions(model):
    noises = []

    n_params = 0
    for name, param in model.named_parameters():
        delta = torch.normal(.0, 1, size=param.size())
        nu = torch.normal(.0, 1, size=param.size())

        param_norm = torch.norm(param)
        delta_norm = torch.norm(delta)
        nu_norm = torch.norm(nu)

        delta /= delta_norm
        delta *= param_norm

        nu /= nu_norm
        nu *= param_norm

        noises.append((delta, nu))

        n_params += np.prod(param.size())

    print(f'A total of {n_params:,} parameters.')

    return noises


def init_network(model, all_noises, alpha, beta):
    with torch.no_grad():
        for param, noises in zip(model.parameters(), all_noises):
            delta, nu = noises
            new_value = param + alpha * delta + beta * nu
            param.copy_(new_value)
    return model
