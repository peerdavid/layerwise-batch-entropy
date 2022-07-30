# coding: utf-8

import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
from torchvision import transforms
from torchvision import datasets


def get_loader(dataset, batch_size, num_workers):
    if dataset == "mnist":
        return get_mnist_loader(batch_size, num_workers)
    elif dataset == "fashionmnist":
        return get_fashionmnist_loader(batch_size, num_workers)
    # elif dataset == "cifar10":
    #     return get_cifar10_loader(batch_size, num_workers)
    # elif dataset == "cifar100":
    #     return get_cifar100_loader(batch_size, num_workers)
    else:
        raise Exception(f"Dataset {dataset} not found.")


def get_mnist_loader(batch_size, num_workers):
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST('.data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [50000, 10000])

    train_eval_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, **train_eval_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, **train_eval_kwargs
    )


    test_dataset = datasets.MNIST('.data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader, test_loader


def get_fashionmnist_loader(batch_size, num_workers):
    transform=transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.FashionMNIST('.data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [50000, 10000])

    train_eval_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, **train_eval_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, **train_eval_kwargs
    )

    test_dataset = datasets.FashionMNIST('.data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader, test_loader


# def get_cifar10_loader(batch_size, num_workers):
#     tmp_set = torchvision.datasets.CIFAR10('.data',
#         train=True,
#         download=True,
#         transform=transforms.Compose([transforms.ToTensor()]))

#     tmp_loader = torch.utils.data.DataLoader(tmp_set, batch_size=len(tmp_set), num_workers=1)
#     tmp_data = next(iter(tmp_loader))
#     mean, std = tmp_data[0].mean(), tmp_data[0].std()

#     train_transform = torchvision.transforms.Compose([
#         torchvision.transforms.RandomCrop(32, padding=4),
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean, std),
#     ])
#     test_transform = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean, std),
#     ])

#     train_dataset = torchvision.datasets.CIFAR10(
#         ".data", train=True, transform=train_transform, download=True)
#     test_dataset = torchvision.datasets.CIFAR10(
#         ".data", train=False, transform=test_transform, download=True)

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True,
#     )
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle=False,
#         pin_memory=True,
#         drop_last=False,
#     )
#     return train_loader, test_loader


# def get_cifar100_loader(batch_size, num_workers):
#     tmp_set = torchvision.datasets.CIFAR100('.data',
#         train=True,
#         download=True,
#         transform=transforms.Compose([transforms.ToTensor()]))

#     tmp_loader = torch.utils.data.DataLoader(tmp_set, batch_size=len(tmp_set), num_workers=1)
#     tmp_data = next(iter(tmp_loader))
#     mean, std = tmp_data[0].mean(), tmp_data[0].std()

#     train_transform = torchvision.transforms.Compose([
#         torchvision.transforms.RandomCrop(32, padding=4),
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean, std),
#     ])
#     test_transform = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean, std),
#     ])

#     train_dataset = torchvision.datasets.CIFAR100(
#         ".data", train=True, transform=train_transform, download=True)
#     test_dataset = torchvision.datasets.CIFAR100(
#         ".data", train=False, transform=test_transform, download=True)

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True,
#     )
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle=False,
#         pin_memory=True,
#         drop_last=False,
#     )
#     return train_loader, test_loader
