from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from batch_entropy import batch_entropy, LBELoss, CELoss
import time
import utils
import copy
from tqdm import tqdm
import numpy as np
import random
from plot import generate_plots

# Very good FAQ for loss surface plots:
# https://losslandscape.com/faq/

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
        x = self.fc_classifier(x)
        return x, a


def train_epoch(args, model, criterion, device, train_loader, optimizer, steps):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        steps += 1

        if steps > args.steps:
            return steps

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, A = model(data)
        loss, _, _ = criterion((output, A), target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            h_a = batch_entropy(A[-1])

            print('Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  H: {:.6f}'.format(
                steps, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), h_a))

    return steps


def evaluate(args, model, criterion, device, dataset_loader):
    model.train()
    ces = []
    acc = []
    lbes = []
    losses = []
    eval_size = 20

    # Evaluate test acc / loss
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataset_loader):
            data, target = data.to(device), target.to(device)
            output, A = model(data)
            loss, ce, lbe = criterion((output, A), target)
            losses.append(loss)
            ces.append(ce)
            lbes.append(lbe)
            pred = output.argmax(dim=1, keepdim=True)
            acc.append(pred.eq(target.view_as(pred)).sum().item() / len(data))

            if len(lbes) > eval_size:
                break

    ces = np.mean(ces)
    acc = np.mean(acc)
    lbes = np.mean(lbes)
    losses = ces+lbes

    return ces, acc, lbes, losses


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Batch Entropy with PyTorch and MNIST')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--steps', type=int, default=10, metavar='N',
                        help='number of steps to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lbe_alpha', type=float, default=0.0,
                    help='Desired entropy at the beginning of trainig.')
    parser.add_argument('--lbe_beta', type=float, default=0.0,
                    help='Weight lbe loss.')
    parser.add_argument('--depth', type=int, default=30,
                        help='Depth of the model')
    parser.add_argument('--width', type=int, default=500,
                        help='Width of the model')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes of the dataset')
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

    steps = 0
    model = FNN(args).to(device)
    criterion = LBELoss(args.depth-2, lbe_alpha=args.lbe_alpha, lbe_beta=args.lbe_beta) if args.lbe_beta != 0.0 else CELoss()
    noises = utils.init_directions(model)
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

            ces = np.empty_like(A)
            accs = np.empty_like(A)
            lbes = np.empty_like(A)
            losses = np.empty_like(A)

            for i in range(RESOLUTION):
                for j in range(RESOLUTION):
                    alpha = A[i, j]
                    beta = B[i, j]
                    net = utils.init_network(load_model(), noises, alpha, beta).to(device)
                    ce, acc, lbe, loss = evaluate(args, net, criterion, device, train_loader)
                    lbes[i, j] = lbe
                    ces[i, j] = ce
                    accs[i, j] = acc
                    losses[i, j] = loss
                    del net
                    print(f'alpha : {alpha:.2f}, beta : {beta:.2f}, ce : {ce:.2f}, lbe : {lbe:.2f}')
                    torch.cuda.empty_cache()

            path = f"./generated/lbe_{args.lbe_beta}/depth_{args.depth}/steps_{steps}"
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(f"{path}/ce.npy", ces)
            np.save(f"{path}/lbe.npy", lbes)
            np.save(f"{path}/loss.npy", losses)
            np.save(f"{path}/acc.npy", accs)
            np.save(f"{path}/X.npy", A)
            np.save(f"{path}/Y.npy", B)

            args.path = path
            print("Generate plots...")
            generate_plots(args)

        #
        # Train one epoch
        #
        steps += train_epoch(args, model, criterion, device, train_loader, optimizer, steps)
        ce, acc, lbe, loss = evaluate(args, model, criterion, device, train_loader)
        print(f"steps={steps} | loss={loss} | lbe={lbe} | ce={ce} | acc={acc}")
        epoch += 1


if __name__ == '__main__':
    main()
