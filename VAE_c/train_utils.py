import os
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from plot import plot_hvf, plot_all_reconstructions
import numpy as np

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 54), reduction='sum')
    # MSE = F.mse_loss(recon_x, x.view(-1, 54), reduction='sum')
    # MAE = F.l1_loss(recon_x, x.view(-1, 54), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return  BCE + KLD
    # return  MAE

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0
    # for batch_idx, (data, _) in enumerate(train_loader):
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    return train_loss / len(train_loader.dataset)

# def test(model, device, test_loader, epoch, results_dir='results_vae'):
#     model.eval()
#     test_loss = 0
#     os.makedirs(results_dir, exist_ok=True)
#     with torch.no_grad():
#         for i, data in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:  # For simplicity, plot only the first batch
#                 # plot_hvf(data[0].cpu().numpy(), recon_batch[0].cpu().numpy(), epoch, i, results_dir)
#                 original_data = data[0].cpu().numpy()
#                 reconstructed_data = recon_batch[0].cpu().numpy()
#                 combined_data = np.stack((original_data, reconstructed_data),
#                                          axis=1)  # Stack along a new axis for side-by-side comparison
#
#                 # Save to text file
#                 np.savetxt(os.path.join(results_dir, f'comparison_data_epoch{epoch}.txt'), combined_data, fmt='%-7.2f',
#                            header='Original  Reconstructed', comments='')
#
#                 plot_hvf(original_data, reconstructed_data, epoch, i, results_dir)
#
#     test_loss /= len(test_loader.dataset)
#     print(f'====> Test set loss: {test_loss:.4f}')


# def test(model, device, test_loader, epoch, results_dir='results_vae_10', num_cases=10):
#     print(f"Testing epoch {epoch}")
#     model.eval()
#     test_loss = 0
#     os.makedirs(results_dir, exist_ok=True)
#     reconstructions = {i: [] for i in range(num_cases)}  # Initialize reconstructions storage
#     originals = None  # Ensure originals are declared outside the loop
#
#     with torch.no_grad():
#         for i, data in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#
#             if i == 0:  # Process only the first batch
#                 if epoch == 10:
#                     originals = data[:num_cases].cpu().numpy()
#                     print(f"Originals captured: {originals.shape}")
#                     print("ttttttttt")
#                 for j in range(num_cases):
#                     reconstructions[j].append(recon_batch[j].cpu().numpy())
#
#     if epoch == 10:  # After last epoch, check if originals are captured and then plot
#         if originals is not None:
#             plot_all_reconstructions(originals, reconstructions, epoch, results_dir)
#         else:
#             print("Error: Original data not captured correctly.")
#
#     test_loss /= len(test_loader.dataset)
#     print(f'====> Test set loss: {test_loss:.4f}')
#
#     return reconstructions, originals

def test(model, device, test_loader, epoch, reconstructions, originals, results_dir='results_vae_10', num_cases=10):
    print(f"Testing epoch {epoch}")
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            if i == 0:  # Process only the first batch
                for j in range(num_cases):
                    reconstructions[j].append(recon_batch[j].cpu().numpy())
                if epoch == 10 and originals is None:  # Capture originals once
                    originals = data[:num_cases].cpu().numpy()
                    print(f"Originals captured: {originals.shape}")

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return originals

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
