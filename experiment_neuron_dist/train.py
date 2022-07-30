from __future__ import print_function
import argparse
import torch
import copy
import torch.nn as nn
from random import randint
import torch.nn.functional as F
import torch.optim as optim
import torch_optimizer as optim_special
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from batch_entropy import batch_entropy, CELoss
import time
import numpy as np
import wandb
import matplotlib.pyplot as plt
import scipy
import math

parser = argparse.ArgumentParser(description='Conflicting bundl--deptes with PyTorch and MNIST')
parser.add_argument('--arch', type=str, default="FNN2", metavar='S',
                    help='Architecture - For WanDB filtering')
parser.add_argument('--depth', type=int, default=6, metavar='S',
                    help='Depth of network')
parser.add_argument('--width', type=int, default=500, metavar='S',
                    help='Width of network')
parser.add_argument('--num_classes', type=int, default=10,
                    help='Number of classes used for predictions')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1024, metavar='N',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--learning_rate', type=float, default=1e-5, metavar='Learning rate',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                    help='Learning rate step gamma (default: 0.5)')
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
n_neurons=5

class FNN(nn.Module):
    def __init__(self, args):
        super(FNN, self).__init__()

        self.args = args
        self.width = args.width
        self.depth = args.depth

        self.fc_in = nn.Linear(28*28, self.width)
        fcs = [nn.Linear(self.width, self.width) for i in range(self.depth-2)]
        self.fcs = nn.ModuleList(fcs)
        self.fc_embeddings = nn.Linear(self.width, self.width)
        self.fc_classifier = nn.Linear(fcs[-1].out_features, args.num_classes)

    def forward(self, x):
        a = []
        x = torch.flatten(x, 1)

        x = F.relu(self.fc_in(x))
        for fc in self.fcs:
            x = F.relu(fc(x))
            a.append(x)

        x = F.relu(self.fc_embeddings(x))
        a.append(x)
        x = self.fc_classifier(x)
        return x, a


def train(args, model, device, train_loader, optimizer, epoch, criterion, seen_samples):
    model.train()
    loss = None

    for batch_idx, (data, target) in enumerate(train_loader):
        start = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output, A = model(data)

        # Add loss if necessary to optimize into the correclbe_alphat direction
        loss, _, _ = criterion((output, A), target)

        loss.backward()
        optimizer.step()
        end = time.time()

        seen_samples += output.shape[0]

        if batch_idx % args.log_interval == 0:

            entropies = [batch_entropy(a) for a in A]
            H_out = entropies[-1]

            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            train_acc = correct / output.shape[0]

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAccuracy: {:.4f}\tLoss: {:.4f}\tTime: {:.4f}\tH_last: {}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_acc, loss.item(), end - start, H_out))

    # Create plot form last batch of epoch
    depth = args.depth
    neuron_a = {}

    # Init random neurons
    for d in range(depth-1):
        random_neurons = [randint(0, args.width-1) for _ in range(n_neurons)]
        neuron_a[d] = {}
        for n in random_neurons:
            vals = A[d][:,n].cpu().detach().numpy()
            neuron_a[d][n] = vals

    return seen_samples, neuron_a

test_acc_sliding = []
def test(model, device, test_loader, criterion, seen_samples):
    global test_acc_sliding
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, A = model(data)
            test_loss += criterion((output, A), target)[0]
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss = test_loss / len(test_loader)
    test_acc = correct / len(test_loader.dataset)

    # Sliding avg of test acc over the last epochs
    test_acc_sliding.append(test_acc)
    test_acc_sliding = test_acc_sliding[-5:]

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc * 100))

    return test_acc


def main():
    # Init dataset
    transform_train=transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    ds_train = datasets.MNIST('.data', train=1, download=True, transform=transform_train)

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

    model = FNN(args=args).to(device)
    criterion = CELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 40],
        gamma=args.gamma)

    # Note that pytorch calls kaiming per default via reset_parameters in __init__:
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L81
    seen_samples = 0
    epoch = 0
    for epoch in range(args.epochs):
        scheduler.step()
        seen_samples, neurons_a = train(args, model, device, train_loader, optimizer, epoch, criterion, seen_samples)
        accuracy = test(model, device, test_loader, criterion, seen_samples)

    plt.clf()
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    fig, axes = plt.subplots(nrows=args.depth-1, ncols=n_neurons, sharex=True, sharey=False)
    fig.tight_layout()
    max_val = max([max(k) for l in neurons_a.values() for k in l.values()])
    min_val = min([min(k) for l in neurons_a.values() for k in l.values()])

    for d in neurons_a.keys():
        for i, n in enumerate(neurons_a[d].keys()):
            ax = axes[d][i]
            data = neurons_a[d][n][:64]
            ax.hist(data, bins=50, range=(min_val, max_val))
            ax.set_yticks([])
            ax.set_xticks([])
            entropy = batch_entropy(torch.tensor([[x] for x in data]))
            title = " H=%.1f" % (entropy)
            ax.set_title(title, size=8)

    fig.savefig(f"neuron_dist.png", bbox_inches='tight', dpi=150)
    return accuracy


if __name__ == '__main__':
    main()