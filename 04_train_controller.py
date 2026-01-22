import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import cma
import gymnasium as gym
import cv2
import time
import os
import sys

# Dodajemy bieżący katalog do ścieżki, żeby widzieć folder src
sys.path.append(os.getcwd())
from src.vae import VAE
from src.rnn import MDNRNN

# --- KONFIGURACJA ---
POPULATION_SIZE = 64  # Liczba równoległych agentów (dostosuj do CPU)
GENERATIONS = 200  # Liczba pokoleń (zwiększyłem, bo robisz fine-tuning)
FRAME_SKIP = 4  # Decyzja co 4 klatki gry
TIME_LIMIT = 1000  # Limit kroków w epizodzie
ENV_NAME = "CarRacing-v3"  # Nowa wersja środowiska Gymnasium
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ścieżki do plików
VAE_PATH = "./vae.pth"
RNN_PATH = "./rnn.pth"
CTRL_PATH = "./controller_best.pth"  # Tu zapisujemy "Mistrza" (zwalidowanego)
BACKUP_PATH = "./controller_backup.pth"  # Tu zapisujemy najlepszego z bieżącej epoki (backup)
LOG_FILE = "./training_log.txt"

# Parametry sieci (muszą pasować do wytrenowanych modeli VAE/RNN)
LATENT_SIZE = 32
HIDDEN_SIZE = 256
ACTION_SIZE = 3
INPUT_SIZE = LATENT_SIZE + HIDDEN_SIZE  # 32 + 256 = 288


# --- PROCES ROBOCZY (CPU) ---
def worker_process(conn):
    """
    Działa w oddzielnym procesie. Obsługuje fizykę gry (Box2D).
    """
    # render_mode="rgb_array" jest wymagany w Gymnasium do pobierania pikseli
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    while True:
        try:
            cmd, data = conn.recv()
        except EOFError:
            break

        if cmd == 'reset':
            # Odbieramy seed od Mastera
            seed = data
            obs, _ = env.reset(seed=seed)

            # --- WAŻNE: Skalowanie obrazka startowego (Naprawa błędu VAE) ---
            if obs is not None:
                obs = obs[:84, :, :]  # Ucięcie paska stanu
                obs = cv2.resize(obs, (64, 64))  # Zmniejszenie do 64x64
            # ----------------------------------------------------------------

            conn.send(obs)

        elif cmd == 'step':
            action = data
            total_reward = 0

            # Frame Skip
            for _ in range(FRAME_SKIP):
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            # Przetwarzanie obrazka
            if obs is not None:
                obs = obs[:84, :, :]
                obs = cv2.resize(obs, (64, 64))

            done = terminated or truncated
            conn.send((obs, total_reward, done))

        elif cmd == 'close':
            env.close()
            break


# --- MASTER (GPU) ---
class ParallelTrainer:
    def __init__(self, pop_size):
        self.pop_size = pop_size
        self.parents = []
        self.processes = []

        print(f"[{time.strftime('%H:%M:%S')}] Uruchamianie {pop_size} workerów...")
        mp.set_start_method('spawn', force=True)

        for _ in range(pop_size):
            parent, child = mp.Pipe()
            p = mp.Process(target=worker_process, args=(child,))
            p.start()
            self.parents.append(parent)
            self.processes.append(p)

        print(f"[{time.strftime('%H:%M:%S')}] Ładowanie modeli VAE i RNN na {DEVICE}...")
        self.vae = VAE(latent_size=LATENT_SIZE).to(DEVICE)
        self.vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
        self.vae.eval()

        self.rnn = MDNRNN(latent_size=LATENT_SIZE, action_size=ACTION_SIZE, hidden_size=HIDDEN_SIZE)
        self.rnn.load_state_dict(torch.load(RNN_PATH, map_location=DEVICE))
        self.rnn.to(DEVICE)
        self.rnn.eval()

    def batched_rollout(self, solutions):
        """
        Wykonuje jedną generację równolegle.
        """
        # 1. Konwersja parametrów na macierze wag
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

        # 2. Synchronizacja toru (Seed)
        # Losujemy JEDEN seed dla całego pokolenia
        current_seed = np.random.randint(0, 100000)

        # Reset workerów
        current_obs = []
        for parent in self.parents:
            parent.send(('reset', current_seed))
        for parent in self.parents:
            current_obs.append(parent.recv())

        # 3. Inicjalizacja RNN
        rnn_hidden = (
            torch.zeros(1, self.pop_size, HIDDEN_SIZE).to(DEVICE),
            torch.zeros(1, self.pop_size, HIDDEN_SIZE).to(DEVICE)
        )

        total_rewards = np.zeros(self.pop_size)
        dones = np.zeros(self.pop_size, dtype=bool)
        step_count = 0

        # 4. Pętla symulacji
        while not all(dones) and step_count < TIME_LIMIT:
            # A. Wizja (VAE)
            obs_batch = np.array(current_obs, dtype=np.float32) / 255.0
            obs_tensor = torch.tensor(obs_batch).permute(0, 3, 1, 2).to(DEVICE)

            with torch.no_grad():
                mu, _ = self.vae.encode(obs_tensor)

            # B. Decyzja (Kontroler)
            h_state = rnn_hidden[0].squeeze(0)
            controller_input = torch.cat([mu, h_state], dim=1).unsqueeze(2)

            # Równoległe obliczenia dla wszystkich agentów
            action_out = torch.bmm(W_batch, controller_input) + b_batch
            action_out = action_out.squeeze(2)

            s = torch.tanh(action_out[:, 0])  # Skręt
            g = torch.sigmoid(action_out[:, 1])  # Gaz
            b = torch.sigmoid(action_out[:, 2])  # Hamulec
            actions = torch.stack([s, g, b], dim=1).cpu().numpy()

            # C. Fizyka
            for i, parent in enumerate(self.parents):
                if not dones[i]:
                    parent.send(('step', actions[i]))

            # D. Odbiór wyników
            for i, parent in enumerate(self.parents):
                if not dones[i]:
                    obs, reward, is_done = parent.recv()
                    total_rewards[i] += reward
                    current_obs[i] = obs

                    if is_done:
                        dones[i] = True
                    # Early stopping (jeśli auto stoi/cofa)
                    if total_rewards[i] < -50:
                        dones[i] = True

            step_count += 1

            # E. Pamięć (RNN)
            z_in = mu.unsqueeze(1)
            a_in = torch.tensor(actions, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            with torch.no_grad():
                _, _, _, rnn_hidden = self.rnn(z_in, a_in, rnn_hidden)

        # CMA minimalizuje, więc zwracamy ujemną nagrodę
        return -total_rewards

    def validate_agent(self, best_params, seeds):
        """
        Sprawdza jednego agenta na zestawie stałych torów (Benchmark).
        Używa jednego workera sekwencyjnie.
        """
        num_weights = ACTION_SIZE * INPUT_SIZE
        w = best_params[:num_weights].reshape(ACTION_SIZE, INPUT_SIZE)
        b = best_params[num_weights:]

        W_tensor = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        b_tensor = torch.tensor(b, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(DEVICE)

        total_score = 0
        worker = self.parents[0]  # Używamy tylko pierwszego procesu

        for seed in seeds:
            worker.send(('reset', seed))
            obs = worker.recv()

            rnn_hidden = (
                torch.zeros(1, 1, HIDDEN_SIZE).to(DEVICE),
                torch.zeros(1, 1, HIDDEN_SIZE).to(DEVICE)
            )

            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < TIME_LIMIT:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0

                with torch.no_grad():
                    mu, _ = self.vae.encode(obs_tensor)
                    h_state = rnn_hidden[0].squeeze(0)
                    controller_input = torch.cat([mu, h_state], dim=1).unsqueeze(2)

                    action_out = torch.bmm(W_tensor, controller_input) + b_tensor
                    action_out = action_out.squeeze(2).squeeze(0)

                    s = torch.tanh(action_out[0])
                    g = torch.sigmoid(action_out[1])
                    b = torch.sigmoid(action_out[2])
                    action = np.array([s.item(), g.item(), b.item()])

                    worker.send(('step', action))
                    obs, reward, done = worker.recv()

                    episode_reward += reward
                    step_count += 1

                    z_in = mu.unsqueeze(1)
                    a_in = torch.tensor(action, dtype=torch.float32).view(1, 1, 3).to(DEVICE)
                    _, _, _, rnn_hidden = self.rnn(z_in, a_in, rnn_hidden)

            total_score += episode_reward

        return total_score / len(seeds)

    def close(self):
        for parent in self.parents:
            parent.send(('close', None))
        for p in self.processes:
            p.join()


def train():
    trainer = ParallelTrainer(POPULATION_SIZE)
    param_count = ACTION_SIZE * INPUT_SIZE + ACTION_SIZE

    # --- WCZYTYWANIE ISTNIEJĄCEGO MODELU (FINE-TUNING) ---
    start_params = np.zeros(param_count)  # Domyślnie zera (gdyby nie było pliku)
    real_best_avg = -float('inf')

    if os.path.exists(CTRL_PATH):
        print(f"\n>>> ZNALEZIONO {CTRL_PATH} - Wczytywanie wag do kontynuacji treningu...")
        try:
            checkpoint = torch.load(CTRL_PATH, map_location=DEVICE)

            # Używamy .data.cpu().numpy() aby uniknąć błędu 'requires_grad'
            w = checkpoint['fc.weight'].data.cpu().numpy().flatten()
            b = checkpoint['fc.bias'].data.cpu().numpy().flatten()

            start_params = np.concatenate([w, b])
            print(">>> Wagi wczytane pomyślnie.")
        except Exception as e:
            print(f"!!! Błąd ładowania wag: {e}. Startuję od zera.")
            start_params = np.zeros(param_count)
    else:
        print("\n>>> Brak zapisanego modelu. Startuję od zera.")

    print(f"Trening na: {DEVICE} | Populacja: {POPULATION_SIZE}")
    print(f"Logi: {LOG_FILE}")

    # Stałe seedy walidacyjne (Trudne tory)
    VALIDATION_SEEDS = [1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10010]

    # --- Initial Validation (Baseline) ---
    # Jeśli wczytaliśmy model, sprawdźmy jak dobry jest na starcie
    if np.any(start_params):
        print(">>> Wykonywanie walidacji startowej na modelu bazowym...")
        real_best_avg = trainer.validate_agent(start_params, seeds=VALIDATION_SEEDS)
        print(f">>> Startowy wynik walidacji (Baseline): {real_best_avg:.1f} pkt")

    # Inicjalizacja CMA-ES
    # sigma=0.1 dla startu od zera, sigma=0.08 dla fine-tuningu (bardziej precyzyjnie)
    initial_sigma = 0.08 if np.any(start_params) else 0.1

    es = cma.CMAEvolutionStrategy(start_params, initial_sigma, {
        'popsize': POPULATION_SIZE,
        'seed': 42
    })

    try:
        for gen in range(GENERATIONS):
            start = time.time()

            # 1. Pobierz populację
            solutions = es.ask()

            # 2. Trening (Losowy tor)
            rewards = trainer.batched_rollout(solutions)
            es.tell(solutions, rewards)

            # Statystyki
            best_train_score = -min(rewards)
            mean_train_score = -np.mean(rewards)
            best_idx = np.argmin(rewards)
            best_params_gen = solutions[best_idx]

            print(f"Gen {gen + 1}: TrainBest={best_train_score:.1f} | Mean={mean_train_score:.1f}", end="")

            # Zawsze zapisz kopię zapasową najlepszego z tej generacji
            dummy = nn.Linear(INPUT_SIZE, ACTION_SIZE)
            w = best_params_gen[:ACTION_SIZE * INPUT_SIZE].reshape(ACTION_SIZE, INPUT_SIZE)
            b = best_params_gen[ACTION_SIZE * INPUT_SIZE:]
            dummy.weight.data = torch.tensor(w, dtype=torch.float32)
            dummy.bias.data = torch.tensor(b, dtype=torch.float32)
            torch.save({'fc.weight': dummy.weight, 'fc.bias': dummy.bias}, BACKUP_PATH)

            # 3. WALIDACJA (tylko obiecujące modele)
            if best_train_score > 100:
                avg_val_score = trainer.validate_agent(best_params_gen, seeds=VALIDATION_SEEDS)
                print(f" | VAL AVG={avg_val_score:.1f}", end="")

                # Nadpisujemy Mistrza TYLKO jeśli wynik jest lepszy niż rekord
                if avg_val_score > real_best_avg:
                    real_best_avg = avg_val_score
                    torch.save({'fc.weight': dummy.weight, 'fc.bias': dummy.bias}, CTRL_PATH)
                    print(" >>> NOWY REKORD (VALIDATED)!", end="")

            print(f" | T={time.time() - start:.1f}s")

            with open(LOG_FILE, "a") as f:
                val_score = avg_val_score if 'avg_val_score' in locals() else 0
                f.write(f"{gen + 1},{best_train_score},{mean_train_score},{val_score}\n")

            if 'avg_val_score' in locals(): del avg_val_score

            if es.stop():
                print("CMA-ES zatrzymany.")
                break

    except KeyboardInterrupt:
        print("\nZatrzymano przez użytkownika.")
    except Exception as e:
        print(f"\nBŁĄD KRYTYCZNY: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.close()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train()