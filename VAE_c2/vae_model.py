import torch
from torch import nn
from torch.nn import functional as F


# class VAE(nn.Module):
#     def __init__(self, latent_dim=10):
#         super(VAE, self).__init__()
#         self.latent_dim = latent_dim
#
#         # Encoder
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # Output: [32, 4, 5]
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: [64, 2, 3]
#         self.fc1 = nn.Linear(64 * 2 * 3, latent_dim)  # Flatten and reduce to latent_dim dimensions
#
#         # Decoder
#         self.fc2 = nn.Linear(latent_dim, 64 * 2 * 3)  # Expand to match the size before the deconv
#         self.conv_transpose1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
#                                                   output_padding=(0, 1))  # Output: [32, 4, 5]
#         self.conv_transpose2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1,
#                                                   output_padding=(0, 1))  # Output: [32, 8, 9]
#         self.conv_transpose3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)  # Output: [1, 8, 9]
#
#     def encode(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
#         return self.fc1(x)  # Returning the latent space directly
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def decode(self, z):
#         z = F.relu(self.fc2(z))
#         z = z.view(-1, 64, 2, 3)
#         z = F.relu(self.conv_transpose1(z))
#         z = F.relu(self.conv_transpose2(z))
#         return torch.sigmoid(self.conv_transpose3(z))  # Sigmoid to bring output to [0, 1] range
#
#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         z_prior = torch.randn_like(z)
#         return self.decode(z), mu, logvar, z, z_prior

import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=10):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # Output: [32, 4, 5]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: [64, 2, 3]
        self.fc1 = nn.Linear(64 * 2 * 3, 256)
        self.fc21 = nn.Linear(256, latent_dim)  # Latent mean
        self.fc22 = nn.Linear(256, latent_dim)  # Latent log variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 64 * 2 * 3)
        # self.conv_transpose1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)  # Output: [32, 4, 6]
        # self.conv_transpose2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)  # Output: [16, 8, 12]
        # self.conv_transpose3 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=1, padding=1,
        #                                           output_padding=0)  # Adjust to [1, 8, 9]

        self.conv_transpose1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)  # Output: [32, 4, 6]
        self.conv_transpose2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)  # Adjust to [16, 8, 12]
        self.conv_transpose3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=(1, 0))  # Adjust to [1, 8, 9]

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)  # Return mu and logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # def decode(self, z):
    #     z = F.relu(self.fc3(z))
    #     z = F.relu(self.fc4(z))
    #     z = z.view(-1, 64, 2, 3)
    #     z = F.relu(self.conv_transpose1(z))
    #     z = F.relu(self.conv_transpose2(z))
    #     return torch.sigmoid(self.conv_transpose3(z))  # Sigmoid to bring output to [0, 1] range

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = z.view(-1, 64, 2, 3)  # Reshape back to the spatial dimensions
        print(f'After first FC layer reshape: {z.shape}')
        z = F.relu(self.conv_transpose1(z))
        print(f'After conv_transpose1: {z.shape}')
        z = F.relu(self.conv_transpose2(z))
        print(f'After conv_transpose2: {z.shape}')
        z = torch.sigmoid(self.conv_transpose3(z))  # Sigmoid to bring output to [0, 1] range
        print(f'Final output after conv_transpose3: {z.shape}')
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z_prior = torch.randn_like(z)
        return self.decode(z), mu, logvar, z, z_prior