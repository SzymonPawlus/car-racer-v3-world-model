import numpy as np
import os


def load_and_normalize_data(human_path, bot_path):
    # 1. Load Human Data (Frame-by-Frame)
    # We must slice it [::4] to match bot speed
    human_files = [os.path.join(human_path, f) for f in os.listdir(human_path) if f.endswith('.npz')]
    processed_episodes = []

    print(f"Processing {len(human_files)} Human episodes (Downsampling 4x)...")
    for f in human_files:
        data = np.load(f)
        obs = data['obs']  # Shape: (T, 64, 64, 3)
        action = data['action']  # Shape: (T, 3)

        # DOWNSAMPLE
        obs = obs[::4]
        action = action[::4]

        processed_episodes.append({'obs': obs, 'action': action})

    # 2. Load Bot Data (Already every 4 frames)
    # Take as is
    bot_files = [os.path.join(bot_path, f) for f in os.listdir(bot_path) if f.endswith('.npz')]

    print(f"Processing {len(bot_files)} Bot episodes (Keep as is)...")
    for f in bot_files:
        data = np.load(f)
        processed_episodes.append({'obs': data['obs'], 'action': data['action']})

    return processed_episodes