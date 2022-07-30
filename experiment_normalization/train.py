from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from typing import Dict
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import torch.optim as optim
import torch_optimizer as optim_special
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from batch_entropy import batch_entropy, LBELoss, CELoss
import time
import numpy as np
import wandb

parser = argparse.ArgumentParser(description='Conflicting bundl--deptes with PyTorch and MNIST')
parser.add_argument('--arch', type=str, default="FNN", metavar='S',
                    help='Architecture - For WanDB filtering')
parser.add_argument('--depth', type=int, default=500, metavar='S',
                    help='Depth of network')
parser.add_argument('--width', type=int, default=1000, metavar='S',
                    help='Width of network')
parser.add_argument("--norm", type=str, default=None, choices=["BatchNorm", "LayerNorm", "WeightNorm", "SELU", "None"], help="Type of normalization to use in each hidden layer")
parser.add_argument('--num_classes', type=int, default=10,
                    help='Number of classes used for predictions')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1024, metavar='N',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--learning_rate', type=float, default=2e-6, metavar='Learning rate',
                    help='learning rate (default: 1.0)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()


# Get device
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device {device}")


class FNN(nn.Module):
    def __init__(self, args):
        super(FNN, self).__init__()

        self.args = args
        self.width = args.width
        self.depth = args.depth
        self.norm = args.norm
        self.activation_fn = F.selu if self.norm == "SELU" else F.relu

        sizes = [28 * 28] + [self.width] * (self.depth - 1)
        layers = []
        for in_features, out_features in zip(sizes[0:-1], sizes[1:]):
            layer: Dict[str, nn.Module] = {
                "fc": nn.Linear(in_features, out_features)
            }
            if self.norm == "BatchNorm":
                layer["norm"] = nn.BatchNorm1d(out_features)
            elif self.norm == "LayerNorm":
                layer["norm"] = nn.LayerNorm(out_features)
            elif self.norm == "WeightNorm":
                layer["fc"] = weight_norm(nn.Linear(in_features, out_features))

            if self.norm == "SELU":
                layer["activation"] = nn.SELU()
            else:
                layer["activation"] = nn.ReLU()

            layers.append(nn.ModuleDict(layer))
        self.layers = nn.ModuleList(layers)
        self.fc_classifier = nn.Linear(self.width, args.num_classes)

    def forward(self, x):
        a = []
        a_pre_norm = []
        h = torch.flatten(x, 1)

        for layer in self.layers:
            h = layer["fc"](h)
            if "norm" in layer:
                a_pre_norm.append(h)
                h = layer["norm"](h)
            h = layer["activation"](h)
            a.append(h)

        y = self.fc_classifier(h)
        return y, a, a_pre_norm


def train(args, model, device, train_loader, optimizer, epoch, criterion, seen_samples):
    model.train()
    Hs = []
    loss = None
    for batch_idx, (data, target) in enumerate(train_loader):
        start = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output, A, A_pre_norm = model(data)

        # Add loss if necessary to optimize into the correct direction
        loss, ce_loss, lbe_loss = criterion((output, A), target)

        loss.backward()
        optimizer.step()
        end = time.time()

        seen_samples += output.shape[0]

        if batch_idx % args.log_interval == 0:

            entropies = [batch_entropy(a) for a in A]
            H_out = entropies[-1]
            H_avg = torch.mean(torch.stack(entropies))
            if len(A_pre_norm) > 0:
                entropies_pre_norm = [batch_entropy(a_pre_norm) for a_pre_norm in A_pre_norm]
                H_out_pre_norm = entropies_pre_norm[-1]
                H_avg_pre_norm = torch.mean(torch.stack(entropies_pre_norm))

            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            train_acc = correct / output.shape[0]

            lbe_alpha_mean = torch.mean(criterion.lbe_alpha_p)
            lbe_alpha_min = torch.min(criterion.lbe_alpha_p)
            lbe_alpha_max = torch.max(criterion.lbe_alpha_p)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAccuracy: {:.4f}\tLoss: {:.4f}\tTime: {:.4f}\tH_last: {}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_acc, loss.item(), end - start, H_out))

            log = {
                "train/h_out": H_out,
                "train/h_avg": H_avg,
                "train/loss": loss,
                "train/loss_lbe": lbe_loss,
                "train/loss_ce": ce_loss,
                "train/accuracy": train_acc,
                "train/lbe_alpha_p": lbe_alpha_mean,
                "train/lbe_alpha_p_min": lbe_alpha_min,
                "train/lbe_alpha_p_max": lbe_alpha_max,
            }
            if len(A_pre_norm) > 0:
                log["train/h_out_pre_norm"] = H_out_pre_norm
                log["train/h_avg_pre_norm"] = H_avg_pre_norm

            wandb.log(log, step=seen_samples)

    return seen_samples

test_acc_sliding = {}
def test(name, model, device, test_loader, criterion, seen_samples):
    global test_acc_sliding
    model.eval()
    test_loss = 0
    correct = 0
    test_acc_sliding[name] = [] if name not in test_acc_sliding else test_acc_sliding[name]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, A, _ = model(data)
            test_loss += criterion((output, A), target)[0]
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = correct / len(test_loader.dataset)

    # Sliding avg of test acc over the last epochs
    test_acc_sliding[name].append(test_acc)
    test_acc_sliding[name] = test_acc_sliding[name][-5:]

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(name,
        test_loss, correct, len(test_loader.dataset), test_acc * 100))

    wandb.log({
        f"{name}/accuracy": np.mean(test_acc_sliding[name]),
        f"{name}/loss_ce": test_loss
    }, step=seen_samples)

    return test_acc


def main():
    # Init dataset
    transform_train=transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    ds_train = datasets.MNIST('.data', train=1, download=True, transform=transform_train)
    ds_train, ds_eval = torch.utils.data.random_split(ds_train, [50000, 10000])

    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    ds_test = datasets.MNIST('.data', train=False, transform=transform_test)

    wandb.init(config=args)

    train_kwargs = {'batch_size': args.batch_size, "shuffle": True}
    test_kwargs = {'batch_size': args.test_batch_size, "shuffle": False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(ds_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(ds_test, **test_kwargs)
    eval_loader = torch.utils.data.DataLoader(ds_eval, **test_kwargs)

    model = FNN(args=args).to(device)

    # criterion = LBELoss(args.depth-2, lbe_alpha=args.lbe_alpha, lbe_beta=args.lbe_beta) if args.lbe_beta != 0.0 else CELoss()
    criterion = CELoss()
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)

    # Note that pytorch calls kaiming per default via reset_parameters in __init__:
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L81
    seen_samples = 0
    epoch = 0
    for epoch in range(args.epochs):
        seen_samples = train(args, model, device, train_loader, optimizer, epoch, criterion, seen_samples)
        if(epoch % 5 == 0):
            accuracy = test("eval", model, device, eval_loader, criterion, seen_samples)
            accuracy = test("test", model, device, test_loader, criterion, seen_samples)
    return accuracy


if __name__ == '__main__':
    main()