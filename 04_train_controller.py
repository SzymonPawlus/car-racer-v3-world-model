import torch
import numpy as np
import cma
import gym
import cv2
import os
import sys
import time
from src.vae import VAE
from src.rnn import MDNRNN
from src.controller import Controller

# --- CONFIGURATION ---
POPULATION_SIZE = 16  # larger = better exploration, slower gen
GENERATIONS = 100
FRAME_SKIP = 4  # Must match your data processing (::4)
TIME_LIMIT = 1000  # Max frames per episode
DEVICE = torch.device("cpu")  # CPU is often faster for sequential gym steps than GPU switching

# Files
VAE_PATH = "./vae.pth"
RNN_PATH = "./rnn.pth"
CTRL_PATH = "./controller_best.pth"
LOG_FILE = "./training_log.txt"


def load_models():
    # Load VAE
    vae = VAE(latent_size=32)
    state_v = torch.load(VAE_PATH, map_location=DEVICE)
    vae.load_state_dict(state_v)
    vae.to(DEVICE)
    vae.eval()

    # Load RNN
    rnn = MDNRNN(latent_size=32, action_size=3, hidden_size=256)
    state_r = torch.load(RNN_PATH, map_location=DEVICE)
    rnn.load_state_dict(state_r)
    rnn.to(DEVICE)
    rnn.eval()

    return vae, rnn


def preprocess(frame):
    # (96, 96, 3) -> (1, 3, 64, 64)
    frame = frame[:84, :, :]
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0
    tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def rollout(params, vae, rnn, env):
    # 1. Load Params into Controller
    controller = Controller(latent_size=32, hidden_size=256, action_size=3)
    torch.nn.utils.vector_to_parameters(torch.tensor(params, dtype=torch.float32), controller.parameters())
    controller.to(DEVICE)
    controller.eval()

    # 2. Reset Env
    obs, _ = env.reset()

    # 3. Init Hidden State
    # LSTM layout: (Layers, Batch, Hidden)
    hidden = (torch.zeros(1, 1, 256).to(DEVICE), torch.zeros(1, 1, 256).to(DEVICE))

    total_reward = 0
    step_count = 0
    done = False

    while not done and step_count < TIME_LIMIT:
        # --- A. VISION ---
        obs_tensor = preprocess(obs)
        with torch.no_grad():
            mu, _ = vae.encode(obs_tensor)  # (1, 32)

        # --- B. CONTROL ---
        h_state = hidden[0].squeeze(0)  # (1, 256)
        action = controller.get_action(mu, h_state)  # numpy [s, g, b]

        # --- C. STEP (Frame Skip) ---
        accumulated_reward = 0
        for _ in range(FRAME_SKIP):
            obs, reward, done, truncated, _ = env.step(action)
            accumulated_reward += reward
            if done or truncated:
                done = True
                break

        # Adjust reward to encourage speed, not just survival
        # CarRacing gives -0.1 every frame. We want to offset that.
        total_reward += accumulated_reward
        step_count += 1

        # Early Stopping for bad agents (spinning or stuck)
        if total_reward < -20:
            done = True

        # --- D. MEMORY UPDATE ---
        # Predict NEXT state to update hidden (context)
        z_in = mu.unsqueeze(1)  # (1, 1, 32)
        a_in = torch.tensor(action, dtype=torch.float32).view(1, 1, 3).to(DEVICE)

        with torch.no_grad():
            _, _, _, hidden = rnn(z_in, a_in, hidden)

    # CMA minimizes, so return negative reward
    return -total_reward


def train():
    # Setup
    env = gym.make("CarRacing-v3", render_mode=None)
    vae, rnn = load_models()

    # Init Controller Params
    dummy = Controller()
    init_params = torch.nn.utils.parameters_to_vector(dummy.parameters()).detach().numpy()
    param_count = len(init_params)
    print(f"Controller Parameters: {param_count}")

    # Init CMA
    # sigma0=0.1 means moderate exploration
    es = cma.CMAEvolutionStrategy(param_count * [0], 0.1, {
        'popsize': POPULATION_SIZE,
        'seed': 42
    })

    print(f"Starting Evolution... Logs in {LOG_FILE}")

    best_ever_reward = -float('inf')

    for gen in range(GENERATIONS):
        start_time = time.time()

        # 1. Get Candidates
        solutions = es.ask()

        # 2. Evaluate (Can be parallelized, but loop is safer for now)
        rewards = []
        for i, sol in enumerate(solutions):
            r = rollout(sol, vae, rnn, env)
            rewards.append(r)
            # print(f"  Agent {i}: {-r:.1f}") # Uncomment for verbose

        # 3. Update CMA
        es.tell(solutions, rewards)

        # 4. Stats
        curr_best = -min(rewards)  # Invert back to positive
        curr_mean = -np.mean(rewards)
        elapsed = time.time() - start_time

        log_str = f"Gen {gen + 1}: Best={curr_best:.1f}, Mean={curr_mean:.1f} [Time: {elapsed:.1f}s]"
        print(log_str)
        with open(LOG_FILE, "a") as f:
            f.write(log_str + "\n")

        # 5. Save if Best
        if curr_best > best_ever_reward:
            best_ever_reward = curr_best
            best_params = es.result.xbest

            # Save logic
            save_c = Controller()
            torch.nn.utils.vector_to_parameters(torch.tensor(best_params, dtype=torch.float32), save_c.parameters())
            torch.save(save_c.state_dict(), CTRL_PATH)
            print(f"  >>> New Best Saved! ({curr_best:.1f})")

        if es.stop():
            print("CMA-ES Converged.")
            break


if __name__ == "__main__":
    train()