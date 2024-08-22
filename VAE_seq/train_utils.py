import os
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from plot import plot_hvf, plot_all_reconstructions
import numpy as np

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # Ensure that the shape of x in loss calculations is properly handled
    # MSE = F.mse_loss(recon_x, x, reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # MAE = F.l1_loss(recon_x, x, reduction='sum')
    return BCE + KLD
    # return MAE

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
    return train_loss / len(train_loader.dataset)


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

# def test(model, device, test_loader, results_dir='results_vae_10', num_cases=10):
#     print("Testing...")
#     model.eval()
#     test_loss = 0
#     case_data = []  # To store (loss, input, target, reconstruction) tuples
#
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             recon_batch, mu, logvar = model(inputs)
#             batch_loss = loss_function(recon_batch, targets, mu, logvar).item()
#             test_loss += batch_loss
#
#             # Save data for each item in the batch
#             for input_sample, target_sample, recon_sample in zip(inputs, targets, recon_batch):
#                 loss_per_sample = loss_function(recon_sample.unsqueeze(0), target_sample.unsqueeze(0), mu, logvar).item()
#                 case_data.append((loss_per_sample, input_sample.cpu().numpy(), target_sample.cpu().numpy(), recon_sample.cpu().numpy()))
#
#     # Now find the top ten cases with the lowest loss
#     case_data.sort(key=lambda x: x[0])  # Sort by loss
#     top_cases = case_data[:num_cases]  # Select top ten cases
#
#     # Separate data for easy access
#     input_o = np.array([x[1] for x in top_cases])
#     originals = np.array([x[2] for x in top_cases])
#     reconstructions = {i: [x[3]] for i, x in enumerate(top_cases)}
#
#     test_loss /= len(test_loader.dataset)
#     print(f'====> Test set loss: {test_loss:.4f}')
#     return originals, input_o, reconstructions





