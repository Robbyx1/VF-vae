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
from train_utils import train, test, sample_from_latent_space, sample_from_latent_space_adjusted, validate, test_and_evaluate
from hvf_dataset import HVFDataset
import random

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

# num_test = int(0.2 * len(full_dataset))  #  20% of the data
# num_train = len(full_dataset) - num_test

num_val = int(0.1 * len(full_dataset))  # 10% for validation
num_test = int(0.1 * len(full_dataset))  # 10% for testing
num_train = len(full_dataset) - num_val - num_test  # Remaining for training

# # Split the dataset
# train_dataset, test_dataset = random_split(full_dataset, [num_train, num_test])
#
# # Setup DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# for i, data in enumerate(train_loader):
#     print(f"Batch {i} shape: {data.shape}")  # data here refers to the batch of images
#     if i == 0:  # Just print the first batch to check
#         break
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [num_train, num_val, num_test])

# Setting up DataLoaders for each dataset split
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)  # No need to shuffle validation data
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Example to show the batch shapes from each DataLoader
for i, data in enumerate(train_loader):
    print(f"Train Batch {i} shape: {data.shape}")
    if i == 0: break  # print just the first batch

for i, data in enumerate(val_loader):
    print(f"Validation Batch {i} shape: {data.shape}")
    if i == 0: break

for i, data in enumerate(test_loader):
    print(f"Test Batch {i} shape: {data.shape}")
    if i == 0: break

# Initialize the model and optimizer
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Main training and testing loop
if __name__ == "__main__":
    originals = []
    reconstructions = {i: [] for i in range(10)}  # Assuming 10 test cases as an example
    results_dir = 'results_vae_11*10'
    os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory is created
    train_losses = []
    validation_losses = []
    latent_mean = 0
    latent_logvar = 0
    latent_std = 0
    total_data_count = 0


    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, args.log_interval, static_mask)
        # originals = test(model, device, test_loader, epoch, reconstructions, originals,static_mask, results_dir)
        val_loss = validate(model, device, val_loader, epoch, static_mask)
        train_losses.append(train_loss)
        validation_losses.append(val_loss)

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

    test_details = test_and_evaluate(model, device, test_loader, static_mask)
    sorted_details = sorted(test_details, key=lambda x: x['loss'])
    best_5 = sorted_details[:5]
    worst_5 = worst_5 = sorted_details[-5:]
    random_5 = random.sample(sorted_details, 5)

    best_5_originals = [entry['original'] for entry in best_5]
    worst_5_originals = [entry['original'] for entry in worst_5]
    random_5_originals = [entry['original'] for entry in random_5]
    best_5_reconstructions = [entry['reconstruction'] for entry in best_5]
    worst_5_reconstructions = [entry['reconstruction'] for entry in worst_5]
    random_5_reconstructions = [entry['reconstruction'] for entry in random_5]

    if best_5_originals is not None:
        save_loss_plot(train_losses,validation_losses, 'Loss/train&valid')
        # plot_all_reconstructions(originals, reconstructions, args.epochs, results_dir)
        plot_comparison(best_5_originals,best_5_reconstructions,full_dataset.mean, full_dataset.std,static_mask, "best_5")
        plot_comparison(worst_5_originals,best_5_reconstructions,full_dataset.mean, full_dataset.std,static_mask, "worst_5")
        plot_comparison(random_5_originals,random_5_reconstructions,full_dataset.mean, full_dataset.std,static_mask, "random_5")


    #     print("Displaying first {} samples:".format(num_samples))
    #     for i in range(num_samples):
    #         print(f"Sample {i + 1}:")
    #         print("Original Data:\n", originals[i])
    #         # print("Original shape:\n", originals[i].shape)
    #         print("Reconstruction:\n", reconstructions[9][i])
    #         # print("Reconstruction shape:\n", reconstructions[9][i].shape)
    #         print("\n")  # Add a newline for better readability between samples
    # else:
    #     print("Error: Original data not captured correctly.")


    # print("Sampled Data Shape:", sampled_data.shape)
    # visualize_latent_space(model, test_loader, device)
    # save_loss_plot(train_losses, 'Loss/train')