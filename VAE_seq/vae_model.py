import torch
from torch import nn
from torch.nn import functional as F

class SequentialVAE(nn.Module):

    def __init__(self, latent_dim=10, intermediate_dim=400):
        super(SequentialVAE, self).__init__()
        self.fc1 = nn.Linear(54*2, intermediate_dim)
        self.fc21 = nn.Linear(intermediate_dim, latent_dim)
        self.fc22 = nn.Linear(intermediate_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, intermediate_dim)
        self.fc4 = nn.Linear(intermediate_dim, 54)

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
        mu, logvar = self.encode(x.view(-1, 108))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
