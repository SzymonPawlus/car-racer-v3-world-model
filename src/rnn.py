import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNRNN(nn.Module):
    def __init__(self, latent_size=32, action_size=3, hidden_size=256, gaussians=5):
        super(MDNRNN, self).__init__()
        self.latent_size = latent_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.gaussians = gaussians

        # Input: z (latent) + a (action)
        self.lstm = nn.LSTM(latent_size + action_size, hidden_size, batch_first=True)

        # Output layers for MDN
        # We need to predict:
        # 1. Pi (mixing coefficients)
        # 2. Mu (means for each gaussian)
        # 3. Sigma (std dev for each gaussian)
        self.fc_pi = nn.Linear(hidden_size, gaussians)
        self.fc_mu = nn.Linear(hidden_size, gaussians * latent_size)
        self.fc_sigma = nn.Linear(hidden_size, gaussians * latent_size)

    def forward(self, z, a, hidden=None):
        """
        z: (Batch, Seq_Len, 32)
        a: (Batch, Seq_Len, 3)
        hidden: (h_0, c_0) tuple or None
        """
        # Concatenate z and a
        # Shape: (Batch, Seq_Len, 35)
        x = torch.cat([z, a], dim=2)

        # Forward LSTM
        output, hidden = self.lstm(x, hidden)

        # Flatten for linear layers
        # Shape: (Batch * Seq_Len, Hidden_Size)
        output_flat = output.contiguous().view(-1, self.hidden_size)

        # Calculate MDN parameters
        # Pi: Softmax ensures they sum to 1 across the gaussians
        pi = F.softmax(self.fc_pi(output_flat), dim=1)

        # Mu: No activation (means can be anything)
        mu = self.fc_mu(output_flat)

        # Sigma: Exp ensures positive standard deviation
        sigma = torch.exp(self.fc_sigma(output_flat))

        # Reshape to separate Gaussians and Latents
        # Shapes become: (Batch, Seq_Len, Gaussians, Latent_Size) (except Pi)
        batch_size = z.size(0)
        seq_len = z.size(1)

        pi = pi.view(batch_size, seq_len, self.gaussians)
        mu = mu.view(batch_size, seq_len, self.gaussians, self.latent_size)
        sigma = sigma.view(batch_size, seq_len, self.gaussians, self.latent_size)

        return pi, mu, sigma, hidden

    def get_loss(self, z_target, pi, mu, sigma):
        """
        Calculates Negative Log Likelihood (NLL).
        z_target: The ACTUAL next latent vector (Batch, Seq, 32)
        """
        z_target = z_target.unsqueeze(2)  # (Batch, Seq, 1, 32)

        # Gaussian Log Likelihood:
        # log(N(x)) = -0.5 * log(2pi) - log(sigma) - 0.5 * ((x-mu)/sigma)^2
        log_scale = torch.log(sigma)
        normal_dist = -0.5 * torch.log(torch.tensor(2 * 3.14159265359)) - log_scale - 0.5 * ((z_target - mu) ** 2) / (
                    sigma ** 2)

        # Sum over latent dimensions (32)
        # Assuming diagonal covariance, log prob of vector is sum of log probs of elements
        log_prob = torch.sum(normal_dist, dim=3)  # (Batch, Seq, Gaussians)

        # Weighted sum using Mixing Coefficients (Pi)
        # LogSumExp trick for numerical stability: log(sum(exp(x)))
        # We want log( sum( pi * N(x) ) )
        # = log( sum( exp( log(pi) + log(N(x)) ) ) )
        log_pi = torch.log(torch.clamp(pi, min=1e-8))

        # Final calculation
        loss = -torch.logsumexp(log_pi + log_prob, dim=2)

        return torch.mean(loss)