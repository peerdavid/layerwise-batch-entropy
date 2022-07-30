from __future__ import print_function
import argparse
from email.mime import base
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional import ssim as compute_ssim
import time
import copy
from tqdm import tqdm
import numpy as np
import random

from batch_entropy import batch_entropy, LBELoss, NoLBELoss
from plot import generate_plots
from autoencoder import AE

# Very good FAQ for loss surface plots:
# https://losslandscape.com/faq/

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



def train_epoch(args, model, criterion, device, train_loader, optimizer, steps):
    model.train()
    for batch_idx, (x, _) in enumerate(train_loader):
        steps += 1

        if steps > args.steps:
            return steps

        x = x.to(device)
        optimizer.zero_grad()

        output = model(x)
        
        x_hat = output["x_hat"]
        A = output["A"]
        loss, _, _ = criterion((x_hat, A), x)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            h_a = batch_entropy(A[-1])

            print('Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  H: {:.6f}'.format(
                steps, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), h_a))

    return steps


def evaluate(args, model, criterion, device, dataset_loader):
    model.train()
    mses = []
    ssim = []
    lbes = []
    losses = []
    eval_size = 20

    # Evaluate test acc / loss
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dataset_loader):
            x = x.to(device)

            output = model(x)

            x_hat = output["x_hat"]
            A = output["A"]
            loss, mse, lbe_loss = criterion((x_hat, A), x)
            losses.append(loss)
            mses.append(mse)
            lbes.append(lbe_loss)
            ssim.append(compute_ssim(x_hat, x).item())

            if len(lbes) > eval_size:
                break

    mses = np.mean(mses)
    ssim = np.mean(ssim)
    lbes = np.mean(lbes)
    losses = mses+lbes

    return mses, ssim, lbes, losses


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Batch Entropy with PyTorch and MNIST')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--steps', type=int, default=10, metavar='N',
                        help='number of steps to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lbe_beta', type=float, default=0.0,
                        help='LBE beta value')
    parser.add_argument('--lbe_alpha', type=float, default=0.5)
    parser.add_argument('--lbe_alpha_min', type=float, default=1.5)
    parser.add_argument('--depth', type=int, default=25)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=10)
    parser.add_argument('--resolution', type=int, default=10, metavar='N',
                        help='Resolution of loss plot')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device %s" % device)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    ds_train = datasets.MNIST('.data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(ds_train,**train_kwargs)

    # Compute entropy of data
    # print("Compute entropy of data.")
    # entropy_data = []
    # for i, (data, _) in enumerate(train_loader):
    #     entropy_data.append(batch_entropy(data))
    #     if(i > 10):
    #         break

    # lbe_alpha = np.mean(entropy_data)
    # print(f"Set lbe_alpha to {lbe_alpha}")

    steps = 0
    model = AE(width=args.width, depth=args.depth, latent_size=args.latent_size).to(device)
    num_layers = len(model.encoder.fcs) + len(model.decoder.fcs)
    if args.lbe_beta != 0.0:
        criterion = LBELoss(nn.MSELoss(), num_layers,lbe_alpha=args.lbe_alpha, lbe_beta=args.lbe_beta, lbe_alpha_min=args.lbe_alpha_min)
    else:
        criterion= NoLBELoss(nn.MSELoss())
    noises = init_directions(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epoch = 0
    while(steps < args.steps):
        #
        # Plot loss surface
        #
        if epoch % 1 == 0:
            def load_model():
                return copy.deepcopy(model)

            RESOLUTION = args.resolution
            A, B = np.meshgrid(np.linspace(-1, 1, RESOLUTION),
                            np.linspace(-1, 1, RESOLUTION), indexing='ij')

            mses = np.empty_like(A)
            ssims = np.empty_like(A)
            lbes = np.empty_like(A)
            losses = np.empty_like(A)

            for i in range(RESOLUTION):
                for j in range(RESOLUTION):
                    alpha = A[i, j]
                    beta = B[i, j]
                    net = init_network(load_model(), noises, alpha, beta).to(device)
                    mse, ssim, lbe, loss = evaluate(args, net, criterion, device, train_loader)
                    lbes[i, j] = lbe
                    mses[i, j] = mse
                    ssims[i, j] = ssim
                    losses[i, j] = loss
                    del net
                    print(f'alpha : {alpha:.2f}, beta : {beta:.2f}, mse : {mse:.2f}, lbe : {lbe:.2f}')
                    torch.cuda.empty_cache()

            path = f"./generated/lbe_{args.lbe_beta}/depth_{args.depth}/steps_{steps}"
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(f"{path}/mse.npy", mses)
            np.save(f"{path}/lbe.npy", lbes)
            np.save(f"{path}/loss.npy", losses)
            np.save(f"{path}/ssim.npy", ssims)
            np.save(f"{path}/X.npy", A)
            np.save(f"{path}/Y.npy", B)

            args.path = path
            print("Generate plots...")
            generate_plots(args)

        #
        # Train one epoch
        #
        steps += train_epoch(args, model, criterion, device, train_loader, optimizer, steps)
        mse, ssim, lbe, loss = evaluate(args, model, criterion, device, train_loader)
        print(f"steps={steps} | loss={loss} | lbe={lbe} | ce={mse} | acc={ssim}")
        epoch += 1


if __name__ == '__main__':
    main()
