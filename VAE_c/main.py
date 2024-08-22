from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from plot import plot_hvf, plot_all_reconstructions, plot_samples, visualize_latent_space, save_loss_plot, plot_comparison
import os
from config import init_mask
from vae_model import VAE
from train_utils import train, test, sample_from_latent_space, sample_from_latent_space_adjusted
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
static_mask = init_mask(device)

# Load the full dataset
full_dataset = HVFDataset('../src/uwhvf/alldata.json')

# Define the size of your test set
num_test = int(0.2 * len(full_dataset))  #  20% of the data
num_train = len(full_dataset) - num_test

# Split the dataset
train_dataset, test_dataset = random_split(full_dataset, [num_train, num_test])

# Setup DataLoaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
for i, data in enumerate(train_loader):
    print(f"Batch {i} shape: {data.shape}")  # data here refers to the batch of images
    if i == 0:  # Just print the first batch to check
        break
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize the model and optimizer
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Main training and testing loop
if __name__ == "__main__":
    originals = None
    reconstructions = {i: [] for i in range(10)}  # Assuming 10 test cases as an example
    results_dir = 'results_vae_11*10'
    os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory is created
    train_losses = []
    latent_mean = 0
    latent_logvar = 0
    latent_std = 0
    total_data_count = 0


    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, args.log_interval, static_mask)
        originals = test(model, device, test_loader, epoch, reconstructions, originals,static_mask, results_dir)
        train_losses.append(train_loss)

        # Reset the sum accumulators and the count at the start of each epoch
        latent_mean = 0
        latent_logvar = 0
        total_data_count = 0

        for _, inputs in enumerate(train_loader):
            inputs = inputs.to(device)
            _, mu, logvar = model(inputs)
            latent_mean += mu.sum(0)
            latent_logvar += logvar.sum(0)
            total_data_count += inputs.size(0)

        # Calculating the mean and standard deviation after processing all batches
        if epoch == args.epochs:  # Only at the last epoch
            latent_mean /= total_data_count
            latent_std = torch.exp(0.5 * latent_logvar / total_data_count)

    if originals is not None:
        save_loss_plot(train_losses, 'Loss/train')
        # plot_all_reconstructions(originals, reconstructions, args.epochs, results_dir)
        plot_comparison(originals,reconstructions,static_mask)
        num_samples = min(5, len(originals), len(reconstructions))

        print("Displaying first {} samples:".format(num_samples))
        for i in range(num_samples):
            print(f"Sample {i + 1}:")
            print("Original Data:\n", originals[i])
            # print("Original shape:\n", originals[i].shape)
            print("Reconstruction:\n", reconstructions[9][i])
            # print("Reconstruction shape:\n", reconstructions[9][i].shape)
            print("\n")  # Add a newline for better readability between samples
    else:
        print("Error: Original data not captured correctly.")


    # print("Sampled Data Shape:", sampled_data.shape)
    # visualize_latent_space(model, test_loader, device)
    # save_loss_plot(train_losses, 'Loss/train')