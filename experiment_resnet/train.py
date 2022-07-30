#!/usr/bin/env python
# coding: utf-8

# From https://github.dev/hysts/pytorch_resnet/blob/master/main.py

from email.policy import default
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
import torchvision.utils

from dataloader import get_loader
from batch_entropy import LBELoss, CELoss, batch_entropy

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0
seen_samples = 0

def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def parse_args():
    parser = argparse.ArgumentParser()

    # data config
    parser.add_argument('--dataset', type=str, default="cifar10")

    # model config
    parser.add_argument('--arch', type=str, default="residual")
    parser.add_argument('--block_type', type=str, default="basic")
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--base_channels', type=int, default=16)

    # run config
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--device', type=str, default="cuda")

    # optim config
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--milestones', type=str, default='[40, 80]')
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lbe_alpha', type=float, default=0.8,
                    help='Desired entropy at the beginning of trainig.')
    parser.add_argument('--lbe_beta', type=float, default=0.0,
                    help='Weight lbe loss.')

    args = parser.parse_args()

    wandb.init(config=args)

    if((args.lbe_alpha == 0 and args.lbe_beta != 0) or (args.lbe_alpha != 0 and args.lbe_beta == 0)):
        wandb.finish(exit_code=0)
        exit()

    if args.block_type == "basic" and args.lbe_beta > 0.0:
        args.block_type = "basic_lbe"

    dataset =args.dataset.lower()
    input_shape = (1, 1, 28, 28) if dataset == "mnist" else \
                  (1, 1, 28, 28) if dataset == "fashionmnist" else \
                  (1, 3, 32, 32) if dataset == "cifar10" else \
                  (1, 3, 32, 32) if dataset == "cifar100" else \
                  (1, 3, 32, 32)
    n_classes = 10 if dataset == "mnist" else \
                10 if dataset == "fashionmnist" else \
                10 if dataset == "cifar10" else \
                100 if dataset == "cifar100" else \
                10

    model_config = OrderedDict([
        ('arch', 'resnet'),
        ('block_type', args.block_type),
        ('depth', args.depth),
        ('base_channels', args.base_channels),
        ('input_shape', input_shape),
        ('n_classes', n_classes),
    ])

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('learning_rate', args.learning_rate),
        ('weight_decay', args.weight_decay),
        ('momentum', args.momentum),
        ('nesterov', args.nesterov),
        ('milestones', json.loads(args.milestones)),
        ('lr_decay', args.lr_decay),
        ('lbe_beta', args.lbe_beta),
        ('lbe_alpha', args.lbe_alpha),
    ])

    data_config = OrderedDict([
        ('dataset', dataset),
    ])

    run_config = OrderedDict([
        ('seed', args.seed),
        ('num_workers', args.num_workers),
    ])

    config = OrderedDict([
        ('model_config', model_config),
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
    ])

    return config


def load_model(config):
    module = importlib.import_module(config['arch'])
    Network = getattr(module, 'Network')
    return Network(config)


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

def train(epoch, model, optimizer, criterion, train_loader):
    global global_step
    global seen_samples

    logger.info('Train {}'.format(epoch))

    model.train()

    ce_meter = AverageMeter()
    lbe_meter = AverageMeter()
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(train_loader):
        global_step += 1
        seen_samples += data.shape[0]

        data = data.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()

        outputs, A = model(data)
        loss, ce_loss, lbe_loss = criterion((outputs, A), targets)
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        accuracy = correct_ / num

        loss_meter.update(loss_, num)
        ce_meter.update(ce_loss.item(), num)
        lbe_meter.update(lbe_loss.item() if hasattr(lbe_loss, "item") else lbe_loss, num)
        accuracy_meter.update(accuracy, num)

        if step % 100 == 0:

            entropies = [batch_entropy(a) for a in A]
            H_out = entropies[-1]
            H_avg = torch.mean(torch.stack(entropies))
            lbe_alpha_mean = torch.mean(criterion.lbe_alpha_p)
            lbe_alpha_min = torch.min(criterion.lbe_alpha_p)
            lbe_alpha_max = torch.max(criterion.lbe_alpha_p)

            wandb.log({
                "train/loss_ce": ce_meter.avg,
                "train/loss_lbe": lbe_meter.avg,
                "train/h_out": H_out,
                "train/h_avg": H_avg,
                "train/loss": loss_meter.avg,
                "train/accuracy": accuracy_meter.avg,
                "train/lbe_alpha_p": lbe_alpha_mean,
                "train/lbe_alpha_p_min": lbe_alpha_min,
                "train/lbe_alpha_p_max": lbe_alpha_max,
            }, step=seen_samples)

            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'Accuracy {:.4f} ({:.4f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            accuracy_meter.val,
                            accuracy_meter.avg,
                        ))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))


test_acc_sliding = {}
def test(name, epoch, model, criterion, test_loader):
    global test_acc_sliding
    test_acc_sliding[name] = [] if name not in test_acc_sliding else test_acc_sliding[name]

    logger.info('{} {}'.format(name, epoch))

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(test_loader):
        data = data.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            outputs, A = model(data)
        loss, _, _ = criterion((outputs, A), targets)

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        loss_meter.update(loss_, num)
        correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(test_loader.dataset)
    test_acc_sliding[name].append(accuracy)
    test_acc_sliding[name] = test_acc_sliding[name][-5:]

    logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch, loss_meter.avg, np.mean(test_acc_sliding[name])))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    wandb.log({
        f"{name}/loss": loss_meter.avg,
        f"{name}/accuracy": np.mean(test_acc_sliding[name]),
    }, step=seen_samples)



def main():
    # parse command line arguments
    config = parse_args()
    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # data loaders
    train_loader, eval_loader, test_loader = get_loader(
        data_config['dataset'],
        optim_config['batch_size'],
        run_config['num_workers'])

    # model
    model = load_model(config['model_config'])
    model.cuda()
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))

    lbe_alpha = optim_config["lbe_alpha"]
    lbe_beta = optim_config["lbe_beta"]
    num_layers = (len(model.stage1) + len(model.stage2) + len(model.stage3)) + 2
    criterion = LBELoss(num_layers,lbe_alpha=lbe_alpha+0.2, lbe_alpha_min=lbe_alpha, lbe_beta=lbe_beta) if lbe_beta != 0.0 else CELoss()
    params = list(model.parameters()) + list(criterion.parameters())

    # optimizer
    optimizer = torch.optim.SGD(
        params,
        lr=optim_config['learning_rate'],
        momentum=optim_config['momentum'],
        weight_decay=optim_config['weight_decay'],
        nesterov=optim_config['nesterov'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=optim_config['milestones'],
        gamma=optim_config['lr_decay'])

    # run test before start training
    test("eval", 0, model, criterion, eval_loader)
    test("test", 0, model, criterion, test_loader)

    for epoch in range(1, optim_config['epochs'] + 1):
        scheduler.step()

        train(epoch, model, optimizer, criterion, train_loader)
        test("eval", epoch, model, criterion, eval_loader)
        test("test", epoch, model, criterion, test_loader)


if __name__ == '__main__':
    main()
