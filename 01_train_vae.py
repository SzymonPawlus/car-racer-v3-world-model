import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
from src.vae import VAE

# --- CONFIGURATION ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-4  # Lower LR is often more stable for VAEs
EPOCHS = 20  # VAEs need time to settle
HUMAN_DATA_DIR = "./data/human"
BOT_DATA_DIR = "./data/bot"
MODEL_SAVE_PATH = "./vae.pth"

# Check for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")


# --- DATASET CLASS ---
class CarRacingDataset(Dataset):
    def __init__(self, human_dir, bot_dir):
        self.images = []
        self.load_data(human_dir, is_human=True)
        self.load_data(bot_dir, is_human=False)

        # Convert list to numpy array for efficient indexing
        # Shape: (N, 64, 64, 3)
        self.images = np.array(self.images, dtype=np.float32) / 255.0

        # PyTorch expects (N, C, H, W)
        self.images = np.transpose(self.images, (0, 3, 1, 2))

        print(f"Dataset Loaded. Total Frames: {len(self.images)}")
        print(f"Data Shape: {self.images.shape}")

    def load_data(self, data_dir, is_human):
        if not os.path.exists(data_dir):
            print(f"Warning: Directory {data_dir} not found.")
            return

        files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        print(f"Loading {len(files)} episodes from {data_dir}...")

        for file_name in files:
            file_path = os.path.join(data_dir, file_name)
            try:
                data = np.load(file_path)
                obs = data['obs']  # Raw images usually (T, 96, 96, 3)

                # 1. Handling Frame Rate Mismatch
                # If human data, take every 4th frame to match bot physics speed
                if is_human:
                    obs = obs[::4]

                for frame in obs:
                    processed_frame = self.preprocess(frame)
                    self.images.append(processed_frame)

            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    def preprocess(self, frame):
        # Frame is likely 96x96x3 (standard Gym)
        # 1. Crop bottom 12 pixels (Score bar/Indicators) -> 84x96
        # This removes the "jittery" numbers that confuse VAEs
        frame = frame[:84, :, :]

        # 2. Resize to 64x64
        # cv2.resize expects (W, H)
        frame = cv2.resize(frame, (64, 64))

        return frame

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx], dtype=torch.float32)


# --- LOSS FUNCTION ---
def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generated image
    x: original image
    mu: latent mean
    logvar: latent log variance
    """
    # 1. Reconstruction Loss (MSE) - "Does it look like the input?"
    # Using sum instead of mean to keep magnitude comparable to KL
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # 2. KL Divergence - "Is the latent space a smooth normal distribution?"
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


# --- TRAINING LOOP ---
def train():
    # 1. Prepare Data
    dataset = CarRacingDataset(HUMAN_DATA_DIR, BOT_DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Prepare Model
    vae = VAE(img_channels=3, latent_size=32).to(DEVICE)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)

    print("Starting VAE Training...")

    for epoch in range(EPOCHS):
        vae.train()
        total_loss = 0

        for batch_idx, data in enumerate(dataloader):
            data = data.to(DEVICE)

            # Forward pass
            recon_batch, mu, logvar = vae(data)

            # Loss
            loss = loss_function(recon_batch, data, mu, logvar)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1} [{batch_idx * len(data)}/{len(dataset)}] Loss: {loss.item() / len(data):.4f}")

        avg_loss = total_loss / len(dataset)
        print(f"====> Epoch: {epoch + 1} Average Loss: {avg_loss:.4f}")

    # 3. Save Model
    torch.save(vae.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()