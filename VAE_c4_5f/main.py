from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.utils import save_image
from plot import plot_hvf, plot_all_reconstructions, plot_samples, visualize_latent_space, save_loss_plot, plot_comparison
import os
from config import init_mask
from vae_model import VAE
from train_utils import train, sample_from_latent_space, sample_from_latent_space_adjusted, validate, test_and_evaluate
from hvf_dataset import HVFDataset
import random
from sklearn.model_selection import KFold

# Setup command line arguments
parser = argparse.ArgumentParser(description='VAE HVF Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--latent-dim', type=int, default=10, metavar='N',
                    help='size of the latent dimension (default: 20)')
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

num_test = int(0.1 * len(full_dataset))  # 10% for testing
num_train = len(full_dataset) - num_test
train_dataset, test_dataset = random_split(full_dataset, [num_train, num_test])

# Setting up the test DataLoader (held-out test set)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
# Perform 5-fold cross-validation on the train_dataset
kf = KFold(n_splits=5, shuffle=True)

# Cross-validation loop
best_val_loss = float('inf')
best_model_path = None
fold_num = 0
for train_idx, val_idx in kf.split(train_dataset):
    fold_num += 1
    print(f"Starting fold {fold_num}...")

    # Subset the train_dataset for the current fold
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    # Create DataLoaders for the current fold
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model and optimizer for this fold
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train for the specified number of epochs
    train_losses = []
    validation_losses = []
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} for fold {fold_num}...")
        train_loss = train(model, device, train_loader, optimizer, epoch, args.log_interval, static_mask)
        val_loss = validate(model, device, val_loader, epoch, static_mask)
        train_losses.append(train_loss)
        validation_losses.append(val_loss)

    # Save the model after training for this fold
    model_save_path = f'model_fold_{fold_num}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save the loss plot for this fold
    title = f'Loss_train_valid_fold_{fold_num}'
    save_loss_plot(train_losses, validation_losses, title)
    if min(validation_losses) < best_val_loss:
        best_val_loss = min(validation_losses)
        best_model_path = model_save_path
        print(f"New best model found at fold {fold_num} with validation loss: {best_val_loss}")

# visualization
print(f"Loading the best model from {best_model_path}")
best_model = VAE(latent_dim=args.latent_dim).to(device)
best_model.load_state_dict(torch.load(best_model_path))

test_details = test_and_evaluate(best_model, device, test_loader, static_mask)
sorted_details = sorted(test_details, key=lambda x: x['loss'])
best_5 = sorted_details[:5]
worst_5 = sorted_details[-5:]
random_5 = random.sample(sorted_details, 5)

# Prepare for visualization
best_5_originals = [entry['original'] for entry in best_5]
worst_5_originals = [entry['original'] for entry in worst_5]
random_5_originals = [entry['original'] for entry in random_5]
best_5_reconstructions = [entry['reconstruction'] for entry in best_5]
worst_5_reconstructions = [entry['reconstruction'] for entry in worst_5]
random_5_reconstructions = [entry['reconstruction'] for entry in random_5]

# Plot comparisons for best, worst, and random reconstructions
plot_comparison(best_5_originals, best_5_reconstructions, static_mask, "best_5")
plot_comparison(worst_5_originals, worst_5_reconstructions, static_mask, "worst_5")
plot_comparison(random_5_originals, random_5_reconstructions, static_mask, "random_5")