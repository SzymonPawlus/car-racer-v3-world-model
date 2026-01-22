import torch
import numpy as np
import cv2
import os
import sys
from src.vae import VAE
from src.rnn import MDNRNN

# --- CONFIGURATION ---
VAE_PATH = "./vae.pth"
RNN_PATH = "./rnn.pth"
DATA_DIR = "./data/processed"
DREAM_LENGTH = 30  # How many frames to dream before snapping back to reality
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    print(f"Loading VAE from {VAE_PATH}...")
    vae = VAE(latent_size=32).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    vae.eval()

    print(f"Loading RNN from {RNN_PATH}...")
    rnn = MDNRNN(latent_size=32, action_size=3, hidden_size=256)
    rnn.load_state_dict(torch.load(RNN_PATH, map_location=DEVICE))
    rnn.to(DEVICE)
    rnn.eval()
    return vae, rnn


def main():
    vae, rnn = load_models()

    # 1. Select File
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npz')]
    if not files:
        print("No data found in data/processed")
        return

    print("\n--- Select Episode to Play ---")
    for i, f in enumerate(files):
        print(f"[{i}] {f}")

    try:
        choice = int(input("Choice: "))
        filename = files[choice]
    except:
        print("Invalid choice.")
        return

    # 2. Load Data
    data = np.load(os.path.join(DATA_DIR, filename))
    mu_full = data['mu']  # (Total_T, 32)
    action_full = data['action']  # (Total_T, 3)
    total_frames = len(mu_full)

    print(f"Loaded {filename}. Length: {total_frames} frames.")
    print(f"Synchronizing every {DREAM_LENGTH} frames.")

    # Setup Window
    window_name = "Top: REALITY | Bottom: DREAM (Auto-Sync)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 600, 600)

    # State Variables
    current_z = None
    hidden = None

    # 3. Main Loop
    for t in range(total_frames - 1):
        # --- LOGIC: SYNC or DREAM? ---
        is_sync_frame = (t % DREAM_LENGTH == 0)

        if is_sync_frame:
            # SNAP BACK TO REALITY
            # We ignore whatever the RNN predicted last step and force the REAL Z
            current_z = torch.tensor(mu_full[t], dtype=torch.float32).view(1, 1, 32).to(DEVICE)
            # Optional: Reset hidden state?
            # Usually better NOT to reset hidden, so it keeps velocity context.
            # But if the dream was very bad, the hidden state might be garbage.
            # Let's keep it to see if it recovers.

        # --- PREPARE INPUTS ---
        # Action taken at time t
        a_in = torch.tensor(action_full[t], dtype=torch.float32).view(1, 1, 3).to(DEVICE)

        # --- RNN STEP ---
        # Predict t+1 from t
        with torch.no_grad():
            pi, mu, sigma, hidden = rnn(current_z, a_in, hidden)

            # Deterministic Selection (Mode)
            best_k = torch.argmax(pi, dim=2).item()
            next_z = mu[0, 0, best_k, :].unsqueeze(0).unsqueeze(0)  # (1, 1, 32)

        # --- VISUALIZATION ---
        with torch.no_grad():
            # Decode the current "Dream" state (next_z)
            img_dream = vae.decode(next_z.squeeze(1)).squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Decode the REAL future (t+1)
            real_next_z = torch.tensor(mu_full[t + 1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            img_real = vae.decode(real_next_z).squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Stack Images
        combined = np.vstack((img_real, img_dream))
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        # Draw Status Borders
        h, w, _ = combined.shape
        half_h = h // 2

        # Top Border (Always Green - Reality)
        cv2.rectangle(combined, (0, 0), (w, half_h), (0, 255, 0), 2)

        # Bottom Border (Green if just synced, Red if dreaming)
        color = (0, 255, 0) if is_sync_frame else (0, 0, 255)
        thickness = 4 if is_sync_frame else 2
        cv2.rectangle(combined, (0, half_h), (w, h), color, thickness)

        # Text Info
        status_text = "SYNC" if is_sync_frame else f"DREAMING (+{t % DREAM_LENGTH})"
        cv2.putText(combined, f"Frame: {t}/{total_frames}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined, status_text, (10, half_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Show Action Bars
        s, g, b = action_full[t]
        # Gas
        cv2.rectangle(combined, (w - 10, h - 10), (w - 5, h - 10 - int(g * 20)), (0, 255, 0), -1)
        # Brake
        cv2.rectangle(combined, (w - 20, h - 10), (w - 15, h - 10 - int(b * 20)), (0, 0, 255), -1)
        # Steer
        cv2.rectangle(combined, (w - 40, h - 5), (w - 40 + int(s * 20), h - 10), (255, 0, 0), -1)

        cv2.imshow(window_name, combined)

        # Advance state
        current_z = next_z

        # Controls
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):  # Pause
            cv2.waitKey(-1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()