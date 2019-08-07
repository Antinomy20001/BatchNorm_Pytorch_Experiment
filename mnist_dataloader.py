import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
import torchvision.transforms


def get_loader(batch_size, num_workers):
#     mean = np.array([0.4914, 0.4822, 0.4465])
#     std = np.array([0.2470, 0.2435, 0.2616])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    dataset_dir = 'datasets'
    train_dataset = torchvision.datasets.MNIST(
        dataset_dir, train=True, transform=train_transform)
    test_dataset = torchvision.datasets.MNIST(
        dataset_dir, train=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader