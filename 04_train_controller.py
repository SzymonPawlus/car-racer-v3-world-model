import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import cma
import gymnasium as gym  # <--- CHANGED
import cv2
import time
import os
from src.vae import VAE
from src.rnn import MDNRNN

# --- CONFIGURATION ---
POPULATION_SIZE = 4
GENERATIONS = 150
FRAME_SKIP = 4
TIME_LIMIT = 1000
ENV_NAME = "CarRacing-v3"   # <--- CHANGE THIS from v2 to v3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Files
VAE_PATH = "./vae.pth"
RNN_PATH = "./rnn.pth"
CTRL_PATH = "./controller_best.pth"
LOG_FILE = "./training_log.txt"

# Controller Definitions
LATENT_SIZE = 32
HIDDEN_SIZE = 256
ACTION_SIZE = 3
INPUT_SIZE = LATENT_SIZE + HIDDEN_SIZE  # 288


# --- WORKER PROCESS ---
def worker_process(conn):
    """
    Runs in a separate process using CPU.
    """
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    while True:
        cmd, data = conn.recv()

        if cmd == 'reset':
            obs, _ = env.reset()

            # --- POPRAWKA: Musimy przeskalować też po resecie! ---
            if obs is not None:
                obs = obs[:84, :, :]  # Ucięcie paska stanu
                obs = cv2.resize(obs, (64, 64))  # Zmniejszenie do 64x64
            # -----------------------------------------------------

            conn.send(obs)

        elif cmd == 'step':
            action = data
            total_reward = 0

            for _ in range(FRAME_SKIP):
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            # --- To już tu było i jest poprawne ---
            if obs is not None:
                obs = obs[:84, :, :]
                obs = cv2.resize(obs, (64, 64))
            # --------------------------------------

            done = terminated or truncated
            conn.send((obs, total_reward, done))

        elif cmd == 'close':
            env.close()
            break

# --- MASTER CLASS ---
class ParallelTrainer:
    def __init__(self, pop_size):
        self.pop_size = pop_size
        self.parents = []
        self.processes = []

        print(f"Spawning {pop_size} worker processes...")
        mp.set_start_method('spawn', force=True)

        for _ in range(pop_size):
            parent, child = mp.Pipe()
            p = mp.Process(target=worker_process, args=(child,))
            p.start()
            self.parents.append(parent)
            self.processes.append(p)

        print(f"Loading Models to {DEVICE}...")
        self.vae = VAE(latent_size=LATENT_SIZE).to(DEVICE)
        self.vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
        self.vae.eval()

        self.rnn = MDNRNN(latent_size=LATENT_SIZE, action_size=ACTION_SIZE, hidden_size=HIDDEN_SIZE)
        self.rnn.load_state_dict(torch.load(RNN_PATH, map_location=DEVICE))
        self.rnn.to(DEVICE)
        self.rnn.eval()

    def batched_rollout(self, solutions):
        # 1. Parse Parameters for Batch Matrix Multiplication
        weights_list = []
        biases_list = []
        num_weights = ACTION_SIZE * INPUT_SIZE

        for sol in solutions:
            w = sol[:num_weights].reshape(ACTION_SIZE, INPUT_SIZE)
            b = sol[num_weights:]
            weights_list.append(w)
            biases_list.append(b)

        W_batch = torch.tensor(np.array(weights_list), dtype=torch.float32).to(DEVICE)
        b_batch = torch.tensor(np.array(biases_list), dtype=torch.float32).unsqueeze(2).to(DEVICE)

        # 2. Reset Envs
        current_obs = []
        for parent in self.parents:
            parent.send(('reset', None))
        for parent in self.parents:
            obs = parent.recv()
            current_obs.append(obs)

        # 3. Init RNN States
        rnn_hidden = (
            torch.zeros(1, self.pop_size, HIDDEN_SIZE).to(DEVICE),
            torch.zeros(1, self.pop_size, HIDDEN_SIZE).to(DEVICE)
        )

        total_rewards = np.zeros(self.pop_size)
        dones = np.zeros(self.pop_size, dtype=bool)
        step_count = 0

        # 4. Main Loop
        while not all(dones) and step_count < TIME_LIMIT:
            # --- VISION ---
            obs_batch = np.array(current_obs, dtype=np.float32) / 255.0
            obs_tensor = torch.tensor(obs_batch).permute(0, 3, 1, 2).to(DEVICE)

            with torch.no_grad():
                mu, _ = self.vae.encode(obs_tensor)

            # --- CONTROL ---
            h_state = rnn_hidden[0].squeeze(0)
            controller_input = torch.cat([mu, h_state], dim=1).unsqueeze(2)

            # BMM: (Batch, 3, 288) x (Batch, 288, 1) -> (Batch, 3)
            action_out = torch.bmm(W_batch, controller_input) + b_batch
            action_out = action_out.squeeze(2)

            # Activation: Steer=Tanh, Gas/Brake=Sigmoid
            s = torch.tanh(action_out[:, 0])
            g = torch.sigmoid(action_out[:, 1])
            b = torch.sigmoid(action_out[:, 2])
            actions = torch.stack([s, g, b], dim=1).cpu().numpy()

            # --- STEP ---
            for i, parent in enumerate(self.parents):
                if not dones[i]:
                    parent.send(('step', actions[i]))

            # --- GATHER ---
            for i, parent in enumerate(self.parents):
                if not dones[i]:
                    obs, reward, is_done = parent.recv()
                    total_rewards[i] += reward
                    current_obs[i] = obs

                    if is_done:
                        dones[i] = True
                    if total_rewards[i] < -20:  # Early Kill
                        dones[i] = True

            step_count += 1

            # --- MEMORY UPDATE ---
            z_in = mu.unsqueeze(1)
            a_in = torch.tensor(actions, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            with torch.no_grad():
                _, _, _, rnn_hidden = self.rnn(z_in, a_in, rnn_hidden)

        return -total_rewards

    def close(self):
        for parent in self.parents:
            parent.send(('close', None))
        for p in self.processes:
            p.join()


def train():
    trainer = ParallelTrainer(POPULATION_SIZE)
    param_count = ACTION_SIZE * INPUT_SIZE + ACTION_SIZE
    print(f"Training with Population: {POPULATION_SIZE}")
    print(f"Parameters per Agent: {param_count}")

    es = cma.CMAEvolutionStrategy(param_count * [0], 0.1, {'popsize': POPULATION_SIZE})

    try:
        best_ever = -float('inf')
        for gen in range(GENERATIONS):
            start = time.time()
            solutions = es.ask()
            rewards = trainer.batched_rollout(solutions)
            es.tell(solutions, rewards)

            best_gen = -min(rewards)
            mean_gen = -np.mean(rewards)
            print(f"Gen {gen + 1}: Best={best_gen:.1f} | Mean={mean_gen:.1f} | Time={time.time() - start:.1f}s")

            if best_gen > best_ever:
                best_ever = best_gen
                best_params = es.result.xbest
                dummy = nn.Linear(INPUT_SIZE, ACTION_SIZE)
                w = best_params[:ACTION_SIZE * INPUT_SIZE].reshape(ACTION_SIZE, INPUT_SIZE)
                b = best_params[ACTION_SIZE * INPUT_SIZE:]
                dummy.weight.data = torch.tensor(w, dtype=torch.float32)
                dummy.bias.data = torch.tensor(b, dtype=torch.float32)
                torch.save({'fc.weight': dummy.weight, 'fc.bias': dummy.bias}, CTRL_PATH)
                print(f"  >>> New Best Saved: {best_gen:.1f}")

            if es.stop(): break
    finally:
        trainer.close()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train()