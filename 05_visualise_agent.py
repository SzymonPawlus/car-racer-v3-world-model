import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import cv2
import os
import sys

# Importy lokalne
sys.path.append(os.getcwd())
from src.vae import VAE
from src.rnn import MDNRNN

# --- KONFIGURACJA ---
ENV_NAME = "CarRacing-v3"
VIDEO_FILENAME = "best_lap.mp4"
SEED = 2002  # Seed toru, który chcesz zobaczyć (np. ten z walidacji)
TIME_LIMIT = 1000

# Ścieżki do modeli
VAE_PATH = "vae.pth"
RNN_PATH = "rnn.pth"
CTRL_PATH = "controller_best.pth"

DEVICE = torch.device("cpu")  # Do wizualizacji CPU wystarczy i jest bezpieczniejsze


class WorldModelAgent:
    def __init__(self):
        # 1. Ładowanie VAE
        self.vae = VAE(latent_size=32).to(DEVICE)
        self.vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
        self.vae.eval()

        # 2. Ładowanie RNN
        self.rnn = MDNRNN(latent_size=32, action_size=3, hidden_size=256)
        self.rnn.load_state_dict(torch.load(RNN_PATH, map_location=DEVICE))
        self.rnn.to(DEVICE)
        self.rnn.eval()

        # 3. Ładowanie Kontrolera (Warstwa Liniowa)
        # Odtwarzamy strukturę z treningu
        self.controller = nn.Linear(32 + 256, 3).to(DEVICE)
        ctrl_checkpoint = torch.load(CTRL_PATH, map_location=DEVICE)
        self.controller.weight.data = ctrl_checkpoint['fc.weight']
        self.controller.bias.data = ctrl_checkpoint['fc.bias']
        self.controller.eval()

        # Stan ukryty RNN
        self.reset_rnn()

    def reset_rnn(self):
        self.hidden = (
            torch.zeros(1, 1, 256).to(DEVICE),
            torch.zeros(1, 1, 256).to(DEVICE)
        )

    def get_action(self, obs):
        # Preprocessing obrazka
        # (96, 96, 3) -> crop -> resize -> (64, 64, 3)
        obs = obs[:84, :, :]
        obs = cv2.resize(obs, (64, 64))

        # Konwersja na tensor (Batch=1)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0

        with torch.no_grad():
            # A. VAE Encode
            mu, _ = self.vae.encode(obs_tensor)

            # B. Controller Decision
            h_state = self.hidden[0].squeeze(0)
            inp = torch.cat([mu, h_state], dim=1)

            action_out = self.controller(inp).squeeze(0)

            s = torch.tanh(action_out[0]).item()  # Steering
            g = torch.sigmoid(action_out[1]).item()  # Gas
            b = torch.sigmoid(action_out[2]).item()  # Brake

            action = np.array([s, g, b])

            # C. RNN Update
            z_in = mu.unsqueeze(1)
            a_in = torch.tensor(action, dtype=torch.float32).view(1, 1, 3).to(DEVICE)
            _, _, _, self.hidden = self.rnn(z_in, a_in, self.hidden)

            return action


def draw_hud(frame, action, score, step):
    """Rysuje statystyki na klatce wideo"""
    # Kopiujemy klatkę, żeby nie psuć oryginału w pamięci
    img = frame.copy()

    # Parametry akcji
    steer, gas, brake = action

    # Pasek Sterowania (Steering)
    # Środek to 50 px od lewej, szerokość 100
    cv2.rectangle(img, (20, 350), (120, 370), (50, 50, 50), -1)  # Tło
    center_x = 70
    offset = int(steer * 50)
    cv2.rectangle(img, (center_x, 350), (center_x + offset, 370), (0, 255, 255), -1)  # Pasek
    cv2.putText(img, "STEER", (45, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Pasek Gazu (Gas)
    cv2.rectangle(img, (140, 350), (160, 350 - int(gas * 50)), (0, 255, 0), -1)
    cv2.putText(img, "GAS", (135, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Pasek Hamulca (Brake)
    cv2.rectangle(img, (180, 350), (200, 350 - int(brake * 50)), (0, 0, 255), -1)
    cv2.putText(img, "BRK", (175, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Wynik
    cv2.putText(img, f"Score: {score:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Step: {step}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return img


def main():
    print(f"Ładowanie agenta z {CTRL_PATH}...")
    agent = WorldModelAgent()

    print(f"Uruchamianie symulacji na torze SEED={SEED}...")
    # render_mode="rgb_array" pozwala pobrać klatkę jako macierz numpy
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    obs, _ = env.reset(seed=SEED)

    # Przygotowanie wideo
    # Standardowe Gymnasium CarRacing ma 600x400
    frame_height, frame_width = 400, 600
    out = cv2.VideoWriter(VIDEO_FILENAME, cv2.VideoWriter_fourcc(*'mp4v'), 50, (frame_width, frame_height))

    total_reward = 0
    steps = 0
    done = False

    try:
        while not done and steps < TIME_LIMIT:
            # 1. Decyzja agenta
            action = agent.get_action(obs)

            # 2. Krok symulacji
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # 3. Renderowanie i zapis
            frame = env.render()  # Zwraca pełny obraz (600x400)

            # Konwersja RGB (Gym) -> BGR (OpenCV)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Rysowanie HUD
            frame = draw_hud(frame, action, total_reward, steps)

            out.write(frame)

            steps += 1
            if steps % 100 == 0:
                print(f"Krok {steps}, Wynik: {total_reward:.1f}")

    finally:
        env.close()
        out.release()
        print(f"\nGotowe! Zapisano wideo jako: {VIDEO_FILENAME}")
        print(f"Końcowy wynik: {total_reward:.1f}")


if __name__ == "__main__":
    main()