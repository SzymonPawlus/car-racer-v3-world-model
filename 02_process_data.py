import torch
import numpy as np
import os
import cv2
from src.vae import VAE

# --- CONFIG ---
HUMAN_DIR = "./data/human"
BOT_DIR = "./data/bot"
OUTPUT_DIR = "./data/processed"
VAE_PATH = "./vae.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_frame(frame):
    # Must match VAE training exactly
    frame = frame[:84, :, :]
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0
    return frame


def process_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load VAE
    vae = VAE(latent_size=32).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    vae.eval()

    # --- PROCESS HUMAN DATA ---
    print("Processing Human Data (Downsampling 4x)...")
    files = [f for f in os.listdir(HUMAN_DIR) if f.endswith('.npz')]

    for filename in files:
        data = np.load(os.path.join(HUMAN_DIR, filename))
        obs = data['obs']  # (T, 96, 96, 3)
        action = data['action']  # (T, 3)

        # Downsample: Keep every 4th frame
        obs = obs[::4]
        action = action[::4]

        encode_and_save(vae, obs, action, "human_" + filename)

    # --- PROCESS BOT DATA ---
    print("Processing Bot Data (No Downsampling)...")
    files = [f for f in os.listdir(BOT_DIR) if f.endswith('.npz')]

    for filename in files:
        data = np.load(os.path.join(BOT_DIR, filename))
        obs = data['obs']
        action = data['action']

        encode_and_save(vae, obs, action, "bot_" + filename)


def encode_and_save(vae, obs_sequence, action_sequence, filename):
    mu_list = []
    logvar_list = []

    # Process in batches to avoid VRAM overflow
    batch_size = 64
    total_frames = len(obs_sequence)

    with torch.no_grad():
        for i in range(0, total_frames, batch_size):
            # Prepare batch
            batch_obs = obs_sequence[i: i + batch_size]

            # Preprocess batch
            processed_batch = []
            for frame in batch_obs:
                processed_batch.append(preprocess_frame(frame))

            # Convert to Tensor
            # (Batch, 64, 64, 3) -> (Batch, 3, 64, 64)
            tensor_batch = torch.tensor(np.array(processed_batch), dtype=torch.float32)
            tensor_batch = tensor_batch.permute(0, 3, 1, 2).to(DEVICE)

            # Encode
            mu, logvar = vae.encode(tensor_batch)

            mu_list.append(mu.cpu().numpy())
            logvar_list.append(logvar.cpu().numpy())

    # Concatenate all batches
    mu_all = np.concatenate(mu_list, axis=0)
    logvar_all = np.concatenate(logvar_list, axis=0)

    # Save Z, Action, Reward, Done?
    # For WorldModels, we generally just need Z and Action for RNN training.
    # We save mu and logvar so we can sample z differently during RNN training if we want (augmentation)
    # or just use mu.

    save_path = os.path.join(OUTPUT_DIR, filename)
    np.savez_compressed(save_path, mu=mu_all, logvar=logvar_all, action=action_sequence)

    # print(f"Saved {filename}: {mu_all.shape}")


if __name__ == "__main__":
    process_dataset()