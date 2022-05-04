#Import libraries
import os
import torch
import torchvision
from torchinfo import summary
from torchvision.utils import make_grid
from PIL import Image
import requests
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from CNN import ImageClassifierNet
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np

def show_example(img, label):
    print('Label: {} ({})'.format(dataset.classes[label], label))
    plt.imshow(img.squeeze(), cmap='Greys_r')
    plt.axis(False)
def split_indices(n, val_frac, seed):
    # Determine the size of the validation set
    n_val = int(val_frac * n)
    np.random.seed(seed)
    # Create random permutation between 0 to n-1
    idxs = np.random.permutation(n)
    # Pick first n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, 8).permute(1, 2, 0), cmap='Greys_r')
        break
#Definitions for enabling training on GPU
def get_default_device():
    """Use GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
#to train model
def train_model(n_epochs, model, train_dl, val_dl, loss_fn, opt_fn, lr):
    """
    Trains the model on a dataset.

    Args:
        n_epochs: number of epochs
        model: ImageClassifierNet object
        train_dl: training dataloader
        val_dl: validation dataloader
        loss_fn: the loss function
        opt_fn: the optimizer
        lr: learning rate

    Returns:
        The trained model.
        A tuple of (model, train_losses, val_losses, train_accuracies, val_accuracies)
    """
    # Record these values the end of each epoch
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    ######################
    #   YOUR CODE HERE   #
    ######################
    for i in range(1, n_epochs + 1):
        tr_loss = 0
        opt_fn.zero_grad()
        # get dataset from dataloader
        train_features, train_labels = next(iter(train_dl))
        val_features, val_labels = next(iter(val_dl))
        # get prediction
        train_out = model(train_features)
        val_out = model(val_features)
        # get loss
        train_loss = loss_fn(train_out, train_labels)
        val_loss = loss_fn(val_out, val_labels)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # update weights?
        train_loss.backward()
        opt_fn.step()
        tr_loss = train_loss.item()

    return model, train_losses, val_losses, train_accuracies, val_accuracies
if __name__ == '__main__':
    # Checking if hardware acceleration enabled
    if int(os.environ.get('COLAB_GPU', 0)) > 0:  # os.environ['COLAB_GPU']
        print("*** GPU connected")
    else:
        print("*** No hardware acceleration: change to GPU under Runtime > Change runtime type > Hardware accelerator")

    # Transform to normalize the data and convert to a tensor
    transform = Compose([ToTensor(),
                         Normalize((0.5,), (0.5,))
                         ])

    # Download the data
    dataset = FashionMNIST('MNIST_data/', download=True, train=True, transform=transform)
    #print(dataset.classes)
    #show_example(*dataset[20])
    #show_example(*dataset[20000])
    #Create training and validation dataset
    val_frac = 0.2  # 0.3, 0.1 ## Set the fraction for the validation set
    rand_seed = 201  # to insure same split for every code execution ## Set the random seed

    train_indices, val_indices = split_indices(len(dataset), val_frac, rand_seed)
    print("#samples in training set: {}".format(len(train_indices)))
    print("#samples in validation set: {}".format(len(val_indices)))
    batch_size = 32  # 64 ## Set the batch size
    # Training sampler and data loader
    train_sampler = SubsetRandomSampler(train_indices)
    train_dl = DataLoader(dataset,
                          batch_size,
                          sampler=train_sampler)

    # Validation sampler and data loader
    val_sampler = SubsetRandomSampler(val_indices)
    val_dl = DataLoader(dataset,
                        batch_size,
                        sampler=val_sampler)
    #show_batch(train_dl)

    #Build model
    model = ImageClassifierNet()
    #summary(model, input_size=(batch_size, 1, 28, 28)) #to show # of parameters
    #enable training on gpu
    device = get_default_device()

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    to_device(model, device)
    #train model
    num_epochs = 20  # Max number of training epochs
    loss_fn = nn.CrossEntropyLoss()  # Define the loss function
    opt_fn = torch.optim.Adam(model.parameters(), lr=0.07)  # Select an optimizer
    lr = 0.07  # Set the learning rate
    history = train_model(num_epochs, model, train_dl, val_dl, loss_fn, opt_fn, lr)
    model, train_losses, val_losses, train_accuracies, val_accuracies = history