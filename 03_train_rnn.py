import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from src.rnn import MDNRNN

# --- CONFIGURATION ---
BATCH_SIZE = 16  # Smaller batch size for RNNs is often better
SEQ_LEN = 32  # Sequence length for Backprop Through Time
HIDDEN_SIZE = 256
GAUSSIANS = 5
LATENT_SIZE = 32
ACTION_SIZE = 3
LEARNING_RATE = 1e-3
EPOCHS = 30
DATA_DIR = "./data/processed"  # Where 02_process_data.py saved files
MODEL_SAVE_PATH = "./rnn.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")


# --- DATASET ---
class RNNData(Dataset):
    def __init__(self, data_dir, seq_len):
        self.seq_len = seq_len
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]

        # We load all data into memory for speed (Latent vectors are small)
        self.episodes = []

        print(f"Loading {len(self.files)} processed episodes...")
        for f in self.files:
            try:
                data = np.load(f)
                mu = data['mu']  # (T, 32)
                action = data['action']  # (T, 3)

                # Check consistency
                if len(mu) != len(action):
                    continue

                # We need at least seq_len + 1 frames to create a target
                if len(mu) > seq_len + 1:
                    self.episodes.append({
                        'z': mu,
                        'action': action
                    })
            except Exception as e:
                print(f"Error loading {f}: {e}")

        print(f"Loaded {len(self.episodes)} valid episodes.")

    def __len__(self):
        # We define length as number of episodes * constant
        # This is just to define how many batches per epoch
        return len(self.episodes) * 5

    def __getitem__(self, idx):
        # 1. Pick a random episode
        # (We ignore idx and sample randomly to ensure good mixing)
        ep = np.random.choice(self.episodes)
        z_seq = ep['z']
        a_seq = ep['action']

        # 2. Pick a random start point
        # We need inputs [t] to predict target [t+1]
        max_start = len(z_seq) - self.seq_len - 1
        start = np.random.randint(0, max_start)
        end = start + self.seq_len + 1

        z_slice = z_seq[start:end]  # Length: SEQ_LEN + 1
        a_slice = a_seq[start:end]  # Length: SEQ_LEN + 1

        # Inputs: z_slice[:-1] (Current state)
        # Actions: a_slice[:-1] (Action taken)
        # Targets: z_slice[1:] (Next state)

        inputs_z = torch.tensor(z_slice[:-1], dtype=torch.float32)
        inputs_a = torch.tensor(a_slice[:-1], dtype=torch.float32)
        targets_z = torch.tensor(z_slice[1:], dtype=torch.float32)

        return inputs_z, inputs_a, targets_z


# --- TRAINING LOOP ---
def train():
    # 1. Data
    dataset = RNNData(DATA_DIR, SEQ_LEN)

    # Simple split: we trust the dataset randomizer, so we use the same dataset
    # but monitor loss. For strictness, you could split the file list.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Model
    rnn = MDNRNN(
        latent_size=LATENT_SIZE,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE,
        gaussians=GAUSSIANS
    ).to(DEVICE)

    optimizer = optim.Adam(rnn.parameters(), lr=LEARNING_RATE)

    # 3. Loop
    print("Starting MDN-RNN Training...")

    for epoch in range(EPOCHS):
        rnn.train()
        total_loss = 0

        for batch_i, (z, a, z_next) in enumerate(dataloader):
            z = z.to(DEVICE)  # (Batch, Seq, 32)
            a = a.to(DEVICE)  # (Batch, Seq, 3)
            z_next = z_next.to(DEVICE)  # (Batch, Seq, 32)

            # Forward pass
            # We initialize hidden state to None (zeros) every batch
            # because we are sampling random chunks, not continuous streams
            pi, mu, sigma, _ = rnn(z, a)

            # Loss Calculation
            loss = rnn.get_loss(z_next, pi, mu, sigma)

            # Optimization
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent explosion (common in RNNs)
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

            if batch_i % 50 == 0:
                print(f"Epoch {epoch + 1} [{batch_i}/{len(dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"====> Epoch: {epoch + 1} Average Loss: {avg_loss:.4f}")

        # Save occasionally
        if (epoch + 1) % 5 == 0:
            torch.save(rnn.state_dict(), MODEL_SAVE_PATH)

    # Final Save
    torch.save(rnn.state_dict(), MODEL_SAVE_PATH)
    print(f"Training Complete. Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()