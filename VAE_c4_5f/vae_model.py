import torch
from torch import nn
from torch.nn import functional as F
from config import init_mask


class VAE(nn.Module):
    def __init__(self, latent_dim=10):  # Latent space size as per the paper
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # Output: [32, 6, 6]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: [64, 3, 3]

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 256)  # Compress to 256
        self.fc_latent = nn.Linear(256, latent_dim)  # Latent space (no mean and logvar now)

        # Decoder fully connected layers
        self.fc3 = nn.Linear(latent_dim, 288)  # Expand latent vector to 288
        self.fc4 = nn.Linear(288, 64 * 3 * 3)  # Map back to the correct shape [64, 3, 3]

        # Decoder convolutional transpose layers (Deconvolution layers)
        self.conv_transpose1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)  # Output: [64, 6, 6]
        self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)  # Output: [32, 12, 12]
        self.conv_transpose3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)  # Output: [1, 12, 12]

    def encode(self, x):
        # Encoder: convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the tensor
        x = x.view(x.size(0), -1)  # Flatten from [64, 3, 3] to [64 * 3 * 3 = 256]

        # Fully connected layer for latent space
        h1 = F.relu(self.fc1(x))
        return self.fc_latent(h1)  # Return latent space (no mean and logvar, deterministic)

    def decode(self, z):
        # Decoder: fully connected layers
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))

        # Reshape to match the decoder input [64, 3, 3]
        z = z.view(-1, 64, 3, 3)

        # Convolutional transpose layers
        z = F.relu(self.conv_transpose1(z))  # Output: [64, 6, 6]
        z = F.relu(self.conv_transpose2(z))  # Output: [32, 12, 12]
        z = torch.sigmoid(self.conv_transpose3(z))  # Output: [1, 12, 12]
        mask = init_mask(z.device)

        return z*mask

    def forward(self, x):
        z = self.encode(x)
        z_prior = torch.randn(z.size(0), self.latent_dim).to(z.device)
        return self.decode(z), z, z_prior