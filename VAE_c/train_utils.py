import os
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from plot import plot_hvf, plot_all_reconstructions
from config import init_mask
import numpy as np


# Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logvar, mask):
#     MAE = F.l1_loss(recon_x * mask, x * mask, reduction='sum')
#     # BCE = F.binary_cross_entropy(recon_x * mask, x * mask, reduction='sum')
#     # MAE = F.l1_loss(recon_x, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return  MAE+KLD

def loss_function(recon_x, x, mu, logvar, mask):
    MAE = F.l1_loss(recon_x, x, reduction='none')
    # MAE = criterion(recon_x, x)
    masked_mae = MAE * mask
    total_loss = torch.sum(masked_mae) / torch.sum(mask)
    # BCE = F.binary_cross_entropy(recon_x * mask, x * mask, reduction='sum')
    # MAE = F.l1_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return  total_loss+KLD


def train(model, device, train_loader, optimizer, epoch, log_interval, mask):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, mask)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    return train_loss / len(train_loader.dataset)

def validate(model, device, val_loader, epoch, mask, results_dir='results_vae_validation', num_cases=10):
    print(f"Validation for epoch {epoch}")
    model.eval()
    # validation_losses = []  # Store individual losses for each validation sample
    overall_val_loss = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            batch_loss = 0

            for j in range(data.size(0)):
                # Calculate the loss for each individual sample
                # loss = loss_function(recon_batch[j:j+1], data[j:j+1], mu[j], logvar[j], mask)
                loss = loss_function(recon_batch[j], data[j], mu[j], logvar[j], mask)
                batch_loss += loss.item()
                # validation_losses.append(loss.item())

            overall_val_loss += batch_loss  # Sum up batch loss to calculate overall validation loss

    overall_val_loss /= len(val_loader.dataset)  # Average the validation loss over all samples
    print(f'====> Validation set loss: {overall_val_loss:.4f}')

    return overall_val_loss
# def test(model, device, test_loader, epoch, reconstructions, originals, mask, results_dir='results_vae_5', num_cases=5):
#     print(f"Testing epoch {epoch}")
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, data in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar, mask).item()
#             if i == 0:  # Process only the first batch
#                 for j in range(num_cases):
#                     reconstructions[j].append(recon_batch[j].cpu().numpy())
#                 if epoch == 10 and originals is None:  # Capture originals once
#                     originals = data[:num_cases].cpu().numpy()
#                     print(f"Originals captured: {originals.shape}")
#     test_loss /= len(test_loader.dataset)
#     print(f'====> Test set loss: {test_loss:.4f}')
#     return originals

def test(model, device, test_loader, epoch, reconstructions, originals, mask, results_dir='results_vae_10', num_cases=10):
    print(f"Testing epoch {epoch}")
    model.eval()
    test_losses = []  # Store individual losses for each sample
    overall_test_loss = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            batch_loss = 0

            for j in range(data.size(0)):
                # Calculate the loss for each individual sample
                loss = loss_function(recon_batch[j:j+1], data[j:j+1], mu[j], logvar[j], mask)
                batch_loss += loss.item()
                if i == 0 and j < num_cases:  # Only capture the first batch's first num_cases samples
                    if epoch == 10:  # Assume you want to capture originals only on the last epoch
                        originals.append(data[j].cpu().numpy())
                    reconstructions[j].append(recon_batch[j].cpu().numpy())
                    test_losses.append(loss.item())  # Store individual losses

            overall_test_loss += batch_loss  # Sum up batch loss to calculate overall test loss

    overall_test_loss /= len(test_loader.dataset)  # Average the test loss over all samples
    print(f'====> Test set loss: {overall_test_loss:.4f}')

    if epoch == 10 and originals is None:  # If originals not captured due to some error, log it
        print("Error: Original data not captured correctly.")
    else:
        print(f"Originals captured: {len(originals)} samples")

    # return originals, reconstructions, test_losses
    return originals

def test_and_evaluate(model, device, test_loader, mask):
    model.eval()
    test_details = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            for i in range(data.size(0)):
                loss = loss_function(recon_batch[i:i+1], data[i:i+1], mu[i], logvar[i], mask).item()
                test_details.append({
                    'original': data[i].cpu().numpy(),
                    'reconstruction': recon_batch[i].cpu().numpy(),
                    'loss': loss
                })
    return test_details



def sample_from_latent_space(model, device, num_samples=10):
    model.eval()
    # Generate random latent variables
    latent_dim = 20
    random_latent_vectors = torch.randn(num_samples, latent_dim).to(device)
    with torch.no_grad():
        generated_data = model.decode(random_latent_vectors)
    return generated_data

def sample_from_latent_space_adjusted(model, device, num_samples=10, latent_mean=0, latent_std=1):

    random_latent_vectors = torch.randn(num_samples, model.latent_dim).to(device) *2*latent_std + latent_mean
    # print(random_latent_vectors[:5])
    sampled_data = model.decode(random_latent_vectors)
    return sampled_data
