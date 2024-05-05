import os
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from plot import plot_hvf, plot_all_reconstructions
import numpy as np

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # Ensure that the shape of x in loss calculations is properly handled
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()  # Set the model to training mode
    train_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):  # Note the use of (data, targets)
        data, targets = data.to(device), targets.to(device)  # Move data and targets to the appropriate device

        optimizer.zero_grad()  # Clear the gradients of all optimized variables
        recon_batch, mu, logvar = model(data)  # Forward pass: compute predicted outputs by passing inputs to the model
        loss = loss_function(recon_batch, targets, mu, logvar)  # Calculate the loss
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        train_loss += loss.item()  # Accumulate the training loss
        optimizer.step()  # Perform a single optimization step (parameter update)

        if batch_idx % log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')


# def test(model, device, test_loader, epoch, reconstructions, originals, results_dir='results_vae_10', num_cases=10):
#     print(f"Testing epoch {epoch}")
#     model.eval()
#     test_loss = 0
#
#     with torch.no_grad():
#         for i, data in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#
#             if i == 0:  # Process only the first batch
#                 for j in range(num_cases):
#                     reconstructions[j].append(recon_batch[j].cpu().numpy())
#                 if epoch == 10 and originals is None:  # Capture originals once
#                     originals = data[:num_cases, :54].cpu().numpy()
#                     print(f"Originals captured: {originals.shape}")
#
#     test_loss /= len(test_loader.dataset)
#     print(f'====> Test set loss: {test_loss:.4f}')
#     return originals
def test(model, device, test_loader, epoch, reconstructions, originals,input_o, results_dir='results_vae_10', num_cases=10):
    print(f"Testing epoch {epoch}")
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            recon_batch, mu, logvar = model(inputs)
            test_loss += loss_function(recon_batch, targets, mu, logvar).item()

            if i == 0:  # Process only the first batch
                for j in range(num_cases):
                    reconstructions[j].append(recon_batch[j].cpu().numpy())
                if epoch == 10 and originals is None:  # Capture originals once
                    input_o = inputs[:num_cases, :54].cpu().numpy()  # Take the first 54 elements as "original"
                    originals = targets[:num_cases].cpu().numpy()
                    print(f"Originals captured: {originals.shape}")

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return originals, input_o
