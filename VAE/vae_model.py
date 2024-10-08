import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(54, 400)
        self.fc21 = nn.Linear(400, self.latent_dim)
        self.fc22 = nn.Linear(400, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, 400)
        self.fc4 = nn.Linear(400, 54)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # return torch.tanh(self.fc4(h3))
        return self.fc4(h3)
        # return torch.sigmoid(self.fc4(h3))


    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 54))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
