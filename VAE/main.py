# from __future__ import print_function
# import argparse
# import torch
# import torch.optim as optim
# from torchvision import datasets, transforms
# from vae_model import VAE
# from train_utils import train, test
# from torchvision.utils import save_image da
# import os
#
# parser = argparse.ArgumentParser(description='VAE MNIST Example')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--no-mps', action='store_true', default=False,
#                     help='disables macOS GPU training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# use_mps = not args.no_mps and torch.backends.mps.is_available()
#
# torch.manual_seed(args.seed)
#
# device = torch.device("cuda" if args.cuda else "mps" if use_mps else "cpu")
#
# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=False, **kwargs)
#
# model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
#
# if __name__ == "__main__":
#     for epoch in range(1, args.epochs + 1):
#         train(model, device, train_loader, optimizer, epoch, args.log_interval)
#         test(model, device, test_loader, epoch)
#         with torch.no_grad():
#             sample = torch.randn(64, 20).to(device)
#             sample = model.decode(sample).cpu()
#             save_image(sample.view(64, 1, 28, 28),
#                        f'results/sample_{epoch}.png')
from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import os

from vae_model import VAE
from train_utils import train, test
from hvf_dataset import HVFDataset

# Setup command line arguments
parser = argparse.ArgumentParser(description='VAE HVF Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                    help='disables macOS GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "mps" if use_mps else "cpu")

# Load the full dataset
full_dataset = HVFDataset('../src/uwhvf/alldata.json')

# Define the size of your test set
num_test = int(0.2 * len(full_dataset))  # Let's say 20% of the data
num_train = len(full_dataset) - num_test

# Split the dataset
train_dataset, test_dataset = random_split(full_dataset, [num_train, num_test])

# Setup DataLoaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize the model and optimizer
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Main training and testing loop
if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, args.log_interval)
        test(model, device, test_loader, epoch)
