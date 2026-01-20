import torch
import numpy as np
import cv2
import os
import sys
from src.vae import VAE

# --- CONFIG ---
MODEL_PATH = "./vae.pth"
HUMAN_DIR = "./data/human"
BOT_DIR = "./data/bot"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vae():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)

    vae = VAE(latent_size=32).to(DEVICE)
    vae.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    vae.eval()
    return vae


def list_and_select_file():
    human_files = [f"(Human) {f}" for f in os.listdir(HUMAN_DIR) if f.endswith('.npz')]
    bot_files = [f"(Bot) {f}" for f in os.listdir(BOT_DIR) if f.endswith('.npz')]
    all_files = human_files + bot_files

    if not all_files:
        print("No .npz files found.")
        sys.exit(1)

    print("\n--- Available Episodes ---")
    for i, f in enumerate(all_files):
        print(f"[{i}] {f}")

    while True:
        try:
            choice = input("\nSelect file number: ")
            idx = int(choice)
            if 0 <= idx < len(all_files):
                selected_name = all_files[idx]
                if "(Human)" in selected_name:
                    path = os.path.join(HUMAN_DIR, selected_name.replace("(Human) ", ""))
                else:
                    path = os.path.join(BOT_DIR, selected_name.replace("(Bot) ", ""))
                return path
        except ValueError:
            pass
        print("Invalid selection.")


def preprocess_batch(frames):
    processed = []
    for f in frames:
        f = f[:84, :, :]  # Crop score
        f = cv2.resize(f, (64, 64))
        f = f / 255.0  # Normalize
        processed.append(f)

    data = np.array(processed, dtype=np.float32)
    tensor = torch.tensor(data).permute(0, 3, 1, 2)  # (T, 3, 64, 64)
    return tensor, data


def main():
    vae = load_vae()
    file_path = list_and_select_file()

    print(f"Loading {file_path}...")
    data = np.load(file_path)
    obs = data['obs']

    print("Processing (Deterministic Mode)...")

    chunk_size = 100
    reconstructions = []
    originals = []

    with torch.no_grad():
        for i in range(0, len(obs), chunk_size):
            chunk = obs[i: i + chunk_size]
            input_tensor, input_numpy = preprocess_batch(chunk)
            input_tensor = input_tensor.to(DEVICE)

            # --- MODIFIED SECTION ---
            # Instead of recon, mu, logvar = vae(input_tensor)
            # We explicitly skip the sampling (reparameterize) step

            mu, _ = vae.encode(input_tensor)  # Get the mean (best guess)
            recon = vae.decode(mu)  # Decode the mean directly

            # ------------------------

            recon_np = recon.cpu().permute(0, 2, 3, 1).numpy()

            reconstructions.append(recon_np)
            originals.append(input_numpy)

    full_recon = np.concatenate(reconstructions, axis=0)
    full_orig = np.concatenate(originals, axis=0)

    print("\nStarting Playback...")
    print("Press 'q' to quit.")

    window_name = "Left: Input | Right: VAE (Clean)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 400)

    for i in range(len(full_orig)):
        # Convert RGB to BGR for OpenCV
        img_orig = cv2.cvtColor(full_orig[i], cv2.COLOR_RGB2BGR)
        img_recon = cv2.cvtColor(full_recon[i], cv2.COLOR_RGB2BGR)

        combined = np.hstack((img_orig, img_recon))
        cv2.imshow(window_name, combined)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()