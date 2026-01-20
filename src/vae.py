import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_size=32):
        super(VAE, self).__init__()
        self.latent_size = latent_size

        # --- ENCODER (3 x 64 x 64) ---
        self.enc1 = nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1)  # -> 32x32
        self.enc2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # -> 16x16
        self.enc3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # -> 8x8
        self.enc4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # -> 4x4

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_size)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_size)

        # --- DECODER ---
        self.fc_dec = nn.Linear(latent_size, 256 * 4 * 4)

        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # -> 8x8
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # -> 16x16
        self.dec3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # -> 32x32
        self.dec4 = nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1)  # -> 64x64

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = x.reshape(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.reshape(-1, 256, 4, 4)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = torch.sigmoid(self.dec4(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar