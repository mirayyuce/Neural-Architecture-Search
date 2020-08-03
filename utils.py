from __future__ import absolute_import
import numpy as np
import random
import torch
import copy
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision


def prepare_data(batch_size=128, valid_frac=0.1, manual_seed=0):
    # data augmentation
    n_holes = 1
    length = 16

    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0,translate=(0.125, 0.125)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_transform.transforms.append(Cutout(n_holes=n_holes, length=length))
    

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=train_transform, download=True)
    valid_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=test_transform, download=True)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_frac * num_train))

    np.random.seed(manual_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader, test_loader


def mixup_data(x, y, alpha=1.0, use_cuda=True):

    """
    Compute the mixup data.
    Return mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class Cutout(object):
    """
    Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def GetSubsequentLayers(layer_id, model_descriptor, no_merge_layers=False):
    # utils for building models
    # get subsequent layers in a NN

    if no_merge_layers:

        subsequentlayers = [layer for layer in model_descriptor['layers'] if
                            (layer_id in layer['input']) and layer['type'] != 'merge']

        if len(subsequentlayers) == 1:
            SingleSSL = 1
        else:
            SingleSSL = 0

        # if SingleSSL == 1 and subsequentlayers[0]['type'] =='conv':
        if SingleSSL == 1 and ((subsequentlayers[0]['type'] == 'conv') or (subsequentlayers[0]['type'] == 'dense')):
            IsConv = 1
        else:
            IsConv = 0

    else:
        subsequentlayers = [layer for layer in model_descriptor['layers'] if layer_id in layer['input']]

        if len(subsequentlayers) == 1:
            SingleSSL = 1
        else:
            SingleSSL = 0

        if SingleSSL == 1 and subsequentlayers[0]['type'] == 'conv':
            IsConv = 1
        else:
            IsConv = 0

    return [subsequentlayers, SingleSSL, IsConv]


def ReplaceInput(layers, layer_id2remove, layer_id2add):
    # utils for building models

    for layer in layers:
        layer['input'] = [int(layer_id2add) if x == int(layer_id2remove) else x for x in layer['input']]


def GetUnusedID(model_descriptor):
    # utils for building models
    # return an unused id

    unused_id = np.max([layers['id'] for layers in model_descriptor['layers']]) + 1

    return unused_id


def InheritWeights(old_model, new_model):

    weightspath = 'tempweights/' + str(random.randint(1, 10 ** 12))

    torch.save(old_model.state_dict(), weightspath)

    new_model.load_state_dict(torch.load(weightspath), strict=False)

    return new_model
