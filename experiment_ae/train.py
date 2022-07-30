#!/usr/bin/env python
# coding: utf-8

from configparser import ParsingError
from enum import auto
from json import encoder
import os
import time
import importlib
import json
from collections import OrderedDict
import logging
import argparse
import numpy as np
import random
import wandb

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
from torchmetrics.functional import ssim as compute_ssim

from dataloader import get_loader
from batch_entropy import LBELoss, NoLBELoss, batch_entropy
import autoencoder
from utils import PlotLatentSpace, DEBUG, WANDB_COMMIT, str2bool, ReconstructImages


logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger("AE")

global_step = 0
seen_samples = 0


def parse_args():
    parser = argparse.ArgumentParser()

    # data config
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "fashionmnist"])

    # model config
    parser.add_argument('--arch', type=str, default="AE", choices=['AE', 'DAE', 'CAE', 'CDAE', 'VAE'])
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=10)

    # run config
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--device', type=str, default="cuda")

    # optim config
    parser.add_argument('--criterion', type=str, default="mse", choices=["mse", "bce"])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lbe_beta', type=float, default=0.3, help='Weight lbe loss.')
    parser.add_argument('--lbe_alpha', type=float, default=2.5)
    parser.add_argument('--lbe_alpha_min', type=float, default=0.5)

    # Utils config
    parser.add_argument("--callback_interval", type=int, default=10, help="Callback interval in number of epochs")

    # TensorBoard
    parser.add_argument(
        '--tensorboard', dest='tensorboard', action='store_true')

    args, unknown_args = parser.parse_known_args()

    # If 'noise_scale' is given as unknown arg, parse it
    noise_scale = None
    if len(unknown_args) > 0:
        if not args.arch == "DAE":
            raise ParsingError(f"Can only supply extra cmdline argument for DAE. Unsupported arguments: {unknown_args}")
        if "--noise_scale" not in unknown_args[0]:
            raise ParsingError("Only 'noise_scale', used by DAE, can be defined as extra cmdline argument")
        
        if len(unknown_args) == 1 and "=" in unknown_args[0]:
            noise_scale = float(unknown_args[0].split("=")[-1])
        elif len(unknown_args) == 2:
            noise_scale = float(unknown_args[1])
        else:
            raise ParsingError("Multiple unknown cmdline arguments are given, can only use '--noise_scale VAL' or '--noise_scale=VAL'.")

    if not DEBUG:
        wandb.init(config=args)

    dataset =args.dataset.lower()
    img_size = (1, 28, 28) if dataset == "mnist" else \
                  (1, 28, 28) if dataset == "fashionmnist" else \
                  (3, 32, 32) if dataset == "cifar10" else \
                  (3, 32, 32) if dataset == "cifar100" else \
                  (3, 32, 32)

    model_config = OrderedDict([
        ('arch', args.arch),
        ('depth', args.depth),
        ('width', args.width),
        ('latent_size', args.latent_size),
        ('img_size', img_size),
    ])
    if noise_scale is not None and args.arch == "DAE":
        model_config["noise_scale"] = noise_scale

    optim_config = OrderedDict([
        ('criterion', args.criterion),
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('base_lr', args.base_lr),
        ('weight_decay', args.weight_decay),
        ('lbe_beta', args.lbe_beta),
        ('lbe_alpha', args.lbe_alpha),
        ('lbe_alpha_min', args.lbe_alpha_min)
    ])

    data_config = OrderedDict([
        ('dataset', dataset),
    ])

    run_config = OrderedDict([
        ('seed', args.seed),
        ('num_workers', args.num_workers),
    ])

    utils_config = OrderedDict([
        ("callback_interval", args.callback_interval),
    ])

    config = OrderedDict([
        ('model_config', model_config),
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
        ('utils_config', utils_config)
    ])

    return config


def load_model(config):
    Network = getattr(autoencoder, config["arch"])
    config.pop("arch")
    return Network(**config)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def train(epoch, model, optimizer, criterion, lbe_fn, train_loader):
    global global_step
    global seen_samples

    logger.info('### TRAIN ####')

    is_vae = isinstance(model, autoencoder.VAE)

    model.train()

    criterion_meter = AverageMeter()
    if is_vae:
        kld_meter = AverageMeter()
    lbe_meter = AverageMeter()
    loss_meter = AverageMeter()
    ssim_meter = AverageMeter()
    start = time.time()
    for step, (x, _) in enumerate(train_loader):
        global_step += 1
        seen_samples += x.shape[0]

        x = x.cuda()

        optimizer.zero_grad()

        output = model(x)

        x_hat = output["x_hat"]
        A = output["A"]
        z = output["z"]
        criterion_loss = criterion(x_hat, x)
        base_loss = criterion_loss
        if is_vae:
            kld_loss = output["kld"]
            base_loss += kld_loss
        loss, lbe_loss = lbe_fn(base_loss, A)
        loss.backward()

        optimizer.step()

        loss_ = loss.item()
        ssim = compute_ssim(x_hat, x).item()
        num = x.size(0)

        loss_meter.update(loss_, num)
        criterion_meter.update(criterion_loss.item(), num)
        if is_vae:
            kld_meter.update(kld_loss.item(), num)
        lbe_meter.update(lbe_loss.item() if hasattr(lbe_loss, "item") else lbe_loss, num)
        ssim_meter.update(ssim, num)

        if step % 100 == 0 or step + 1 == len(train_loader):

            H_z = batch_entropy(z)
            entropies = [batch_entropy(a) for a in A]
            H_out = entropies[-1]
            H_avg = torch.mean(torch.stack(entropies))
            lbe_alpha_mean = torch.mean(lbe_fn.lbe_alpha_p)
            lbe_alpha_min = torch.min(lbe_fn.lbe_alpha_p)
            lbe_alpha_max = torch.max(lbe_fn.lbe_alpha_p)

            if not DEBUG:
                logs = {
                    f"train/{criterion.__class__.__name__}": criterion_meter.avg,
                    "train/lbe_loss": lbe_meter.avg,
                    "train/h_z": H_z,
                    "train/h_out": H_out,
                    "train/h_avg": H_avg,
                    "train/loss": loss_meter.avg,
                    "train/ssim": ssim_meter.avg,
                    "train/lbe_alpha_p": lbe_alpha_mean,
                    "train/lbe_alpha_p_min": lbe_alpha_min,
                    "train/lbe_alpha_p_max": lbe_alpha_max,
                }
                if is_vae:
                    logs[f"train/kld_loss"] = kld_meter.avg

                wandb.log(logs, step=seen_samples, commit=WANDB_COMMIT)

            logger.info(f'Epoch {epoch} Step {step}/{len(train_loader) - 1} '
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        f'SSIM {ssim_meter.val:.4f} ({ssim_meter.avg:.4f})')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')


test_ssim_sliding = {}
def evaluate(stage, epoch, model, criterion, lbe_fn, test_loader):
    global test_ssim_sliding
    test_ssim_sliding[stage] = [] if stage not in test_ssim_sliding else test_ssim_sliding[stage]

    logger.info(f'### {stage.upper()} ###')

    is_vae = isinstance(model, autoencoder.VAE)

    model.eval()

    loss_meter = AverageMeter()
    ssim_meter = AverageMeter()
    start = time.time()
    for step, (x, _) in enumerate(test_loader):
        x = x.cuda()
        with torch.no_grad():   
            output = model(x)

        x_hat = output["x_hat"]
        A = output["A"]
        criterion_loss = criterion(x_hat, x)
        base_loss = criterion_loss
        if is_vae:
            kld_loss = output["kld"]
            base_loss += kld_loss
        loss, _ = lbe_fn(base_loss, A)

        loss_ = loss.item()
        ssim = compute_ssim(x_hat, x).item()
        num = x.size(0)

        loss_meter.update(loss_, num)
        ssim_meter.update(ssim, num)

    mean_ssim = ssim_meter.sum / len(test_loader.dataset)
    test_ssim_sliding[stage].append(mean_ssim)
    test_ssim_sliding[stage] = test_ssim_sliding[stage][-5:]

    logger.info(f'Epoch {epoch} Loss {loss_meter.avg:.4f} SSIM {np.mean(test_ssim_sliding[stage]):.4f}')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')

    if not DEBUG:
        wandb.log({
            f"{stage}/loss": loss_meter.avg,
            f"{stage}/SSIM": np.mean(test_ssim_sliding[stage]),
        }, step=seen_samples, commit=WANDB_COMMIT)

    return loss_meter.avg


def main():
    # parse command line arguments
    config = parse_args()
    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']
    utils_config = config['utils_config']

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # data loaders
    train_loader, val_loader, test_loader = get_loader(
        data_config['dataset'],
        optim_config['batch_size'],
        run_config['num_workers'])

    # model
    model = load_model(config['model_config'])
    model.cuda()
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))

    # LBELoss
    lbe_beta = optim_config["lbe_beta"]
    lbe_alpha = optim_config["lbe_alpha"]
    lbe_alpha_min = optim_config["lbe_alpha_min"]
    num_layers = len(model.encoder.fcs) + len(model.decoder.fcs) + 1
    if optim_config["criterion"] == "mse":
        criterion = nn.MSELoss()
    elif optim_config["criterion"] == "bce":
        criterion = nn.BCELoss()
    else:
        raise ValueError(f"Invalid criterion, choose from ['mse', 'bce']")
    if lbe_beta != 0.0:
        lbe_fn = LBELoss(num_layers,lbe_alpha=lbe_alpha, lbe_beta=lbe_beta, lbe_alpha_min=lbe_alpha_min)
    else:
        lbe_fn = NoLBELoss()
    params = list(model.parameters()) + list(lbe_fn.parameters())

    # optimizer
    optimizer = torch.optim.Adam(
        params,
        lr=optim_config['base_lr'],
        weight_decay=optim_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Callbacks
    reconstruct_images_cb = ReconstructImages(
        num_images=5, 
        mean_only=True
    )

    plot_latent_space_cb = PlotLatentSpace(
        num_batches=None,
        quantiles=[0.025, 0.975],
        size_recon=12
    )

    # run validation before start training
    evaluate("val", 0, model, criterion, lbe_fn, test_loader)
    reconstruct_images_cb("val", model, val_loader, seen_samples)
    plot_latent_space_cb("val", model, val_loader, seen_samples)

    if not DEBUG:
        wandb.log({}, step=seen_samples) # Flush wandb logs

    for epoch in range(1, optim_config['epochs'] + 1):
        train(epoch, model, optimizer, criterion, lbe_fn, train_loader)
        
        val_loss = evaluate("val", epoch, model, criterion, lbe_fn, val_loader)

        scheduler.step(val_loss)

        if epoch % utils_config["callback_interval"] == 0:
            reconstruct_images_cb("val", model, val_loader, seen_samples)
            plot_latent_space_cb("val", model, val_loader, seen_samples)

        if not DEBUG:
            wandb.log({}, step=seen_samples) # Flush wandb logs

    # Test model
    evaluate("test", epoch, model, criterion, lbe_fn, test_loader)
    reconstruct_images_cb("test", model, test_loader, seen_samples)
    plot_latent_space_cb("test", model, test_loader, seen_samples)

    if not DEBUG:
        wandb.log({}, step=seen_samples) # Flush wandb logs

        wandb.finish()


if __name__ == '__main__':
    main()
