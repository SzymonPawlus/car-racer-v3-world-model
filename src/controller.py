import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(self, latent_size=32, hidden_size=256, action_size=3):
        super().__init__()
        # Input: concatenation of VAE latent (z) and RNN hidden state (h)
        # z size: 32
        # h size: 256
        self.fc = nn.Linear(latent_size + hidden_size, action_size)

    def forward(self, z, h):
        """
        z: (Batch, 32)
        h: (Batch, 256) - We usually use the hidden state, not cell state
        """
        inp = torch.cat([z, h], dim=1)
        out = self.fc(inp)

        # CarRacing Actions: [Steering, Gas, Brake]
        # Steering: [-1, 1] -> Tanh
        # Gas: [0, 1] -> Sigmoid
        # Brake: [0, 1] -> Sigmoid

        steering = torch.tanh(out[:, 0])
        gas = torch.sigmoid(out[:, 1])
        brake = torch.sigmoid(out[:, 2])

        return torch.stack([steering, gas, brake], dim=1)

    def get_action(self, z, h):
        with torch.no_grad():
            return self.forward(z, h).cpu().numpy().flatten()