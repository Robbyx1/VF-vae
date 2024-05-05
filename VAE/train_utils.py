import os
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from plot import plot_hvf

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 54), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

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

# def test(model, device, test_loader, epoch):
#     model.eval()
#     test_loss = 0
#     results_dir = 'results'
#     os.makedirs(results_dir, exist_ok=True)
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(-1, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                          os.path.join(results_dir, f'reconstruction_{epoch}.png'), nrow=n)
#     test_loss /= len(test_loader.dataset)
#     print(f'====> Test set loss: {test_loss:.4f}')
def test(model, device, test_loader, epoch, results_dir='results_vae'):
    model.eval()
    test_loss = 0
    os.makedirs(results_dir, exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:  # For simplicity, plot only the first batch
                plot_hvf(data[0].cpu().numpy(), recon_batch[0].cpu().numpy(), epoch, i, results_dir)

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
