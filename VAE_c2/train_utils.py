import os
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from plot import plot_hvf, plot_all_reconstructions
from config import init_mask
import numpy as np


# def compute_mmd(z, z_prior, kernel_bandwidth=2.0):
#     # Compute pairwise distances
#     z_dist = torch.cdist(z, z)
#     z_prior_dist = torch.cdist(z_prior, z_prior)
#     cross_dist = torch.cdist(z, z_prior)
#
#     # Apply Gaussian Kernel
#     mmd = torch.exp(-z_dist ** 2 / kernel_bandwidth).mean()
#     mmd += torch.exp(-z_prior_dist ** 2 / kernel_bandwidth).mean()
#     mmd -= 2 * torch.exp(-cross_dist ** 2 / kernel_bandwidth).mean()
#
#     return mmd

def compute_kernel(x, y, sigma_sqr=1):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)

    kernel = torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim)
    return kernel


def compute_mmd(x, y, sigma_sqr=1):
    x_kernel = compute_kernel(x, x, sigma_sqr)
    y_kernel = compute_kernel(y, y, sigma_sqr)
    xy_kernel = compute_kernel(x, y, sigma_sqr)

    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd

def loss_function(recon_x, x, mu, logvar, mask, z, z_prior):
    MAE = F.l1_loss(recon_x, x, reduction='none')
    # MAE = criterion(recon_x, x)
    masked_mae = MAE * mask
    total_loss = torch.sum(masked_mae) / (torch.sum(mask) * x.shape[0])
    # total_loss = torch.sum(masked_mae) / (torch.sum(mask))
    BCE = F.binary_cross_entropy(recon_x * mask, x * mask, reduction='sum')
    # MAE = F.l1_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    mse_loss = F.mse_loss(recon_x, x, reduction='none')

    # Apply the mask to the MSE loss
    masked_mse = mse_loss * mask

    # Calculate the total reconstruction loss
    mse_f = torch.sum(masked_mse) / (torch.sum(mask) * x.shape[0])
    mmd_loss = compute_mmd(z, z_prior)
    # return  BCE+KLD
    return  mse_f + mmd_loss


def train(model, device, train_loader, optimizer, epoch, log_interval, mask):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, z, z_prior = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, mask, z, z_prior)
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
    overall_val_loss = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = data.to(device)
            recon_batch, mu, logvar, z, z_prior = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, mask, z, z_prior)
            overall_val_loss += loss.item()

    overall_val_loss /= len(val_loader.dataset)  # Average over all batches, not samples
    print(f'====> Validation set loss: {overall_val_loss:.4f}')

    return overall_val_loss


def test_and_evaluate(model, device, test_loader, mask):
    model.eval()
    test_details = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar, z, z_prior = model(data)
            for i in range(data.size(0)):
                loss = loss_function(recon_batch[i:i+1], data[i:i+1], mu[i], logvar[i], mask, z[i].unsqueeze(0) , z_prior[i].unsqueeze(0)).item()
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
