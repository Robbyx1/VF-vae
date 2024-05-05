from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from plot import plot_hvf, plot_all_reconstructions
import os

from vae_model import SequentialVAE
from train_utils import train, test
from hvf_dataset import SequentialHVFDataset

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
full_dataset = SequentialHVFDataset('../src/uwhvf/alldata.json')

# Define the size of your test set
num_test = int(0.2 * len(full_dataset))  # Let's say 20% of the data
num_train = len(full_dataset) - num_test

# Split the dataset
train_dataset, test_dataset = random_split(full_dataset, [num_train, num_test])

# Setup DataLoaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize the model and optimizer
model = SequentialVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Main training and testing loop
if __name__ == "__main__":
    originals = None
    input_o = None
    reconstructions = {i: [] for i in range(10)}  # Assuming 10 test cases as an example
    results_dir = 'results_seq_vae_12*10'
    os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory is created

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, args.log_interval)
        originals, input_o = test(model, device, test_loader, epoch, reconstructions, originals,input_o, results_dir)
        # originals, input_o, reconstructions = test(model, device, test_loader, results_dir, num_cases=10)


    if originals is not None:
        plot_all_reconstructions(originals, reconstructions,input_o, args.epochs, results_dir)
    else:
        print("Error: Original data not captured correctly.")