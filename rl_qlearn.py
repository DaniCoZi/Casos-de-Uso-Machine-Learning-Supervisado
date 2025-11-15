import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

class GridWorld:
    def __init__(self, n_rows=5, n_cols=5, max_steps=50):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_steps = max_steps
        self.start = (0, 0)
        self.goal = (n_rows - 1, n_cols - 1)
        self.obstacles = set([])

        self.state = None
        self.steps = 0
        self.n_actions = 4

    @property
    def n_states(self):
        return self.n_rows * self.n_cols

    def _coords_to_state(self, r, c):
        return r * self.n_cols + c

    def _state_to_coords(self, s):
        return s // self.n_cols, s % self.n_cols

    def reset(self):
        self.state = self._coords_to_state(*self.start)
        self.steps = 0
        return self.state

    def step(self, action):
        self.steps += 1
        r, c = self._state_to_coords(self.state)

        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            r = min(self.n_rows - 1, r + 1)
        elif action == 2:
            c = max(0, c - 1)
        elif action == 3:
            c = min(self.n_cols - 1, c + 1)

        reward = -1
        if (r, c) in self.obstacles:
            reward = -5

        if (r, c) == self.goal:
            reward = 10
            done = True
        else:
            done = False

        if self.steps >= self.max_steps:
            done = True

        self.state = self._coords_to_state(r, c)
        return self.state, reward, done, {}

    def greedy_trajectory(self, q_table):
        s = self.reset()
        traj = []
        total_reward = 0

        for _ in range(self.max_steps):
            r, c = self._state_to_coords(s)
            traj.append((r, c))

            a = int(np.argmax(q_table[s]))
            s2, reward, done, _ = self.step(a)
            total_reward += reward
            s = s2
            if done:
                break

        return traj, total_reward


def train_q_learning(
    episodes=300,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    base_path="."
):
    env = GridWorld()
    q_table = np.zeros((env.n_states, env.n_actions))
    epsilon = epsilon_start
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, done, _ = env.step(action)

            q_table[state, action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            total_r += reward
            state = next_state

        rewards.append(total_r)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    traject, demo_reward = env.greedy_trajectory(q_table)

    # --- Guardar resultados ---
    static_dir = os.path.join(base_path, "static")
    models_dir = os.path.join(base_path, "models")
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "q_table_gridworld.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(q_table, f)

    # Gráfica de recompensas
    rewards_img = "rl_rewards.png"
    rp = os.path.join(static_dir, rewards_img)
    plt.figure()
    plt.plot(rewards)
    plt.title("Recompensa por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.tight_layout()
    plt.savefig(rp)
    plt.close()

    # Trayectoria
    traj_img = "rl_trayectoria.png"
    tp = os.path.join(static_dir, traj_img)

    grid = np.zeros((env.n_rows, env.n_cols))
    for i, (r, c) in enumerate(traject):
        grid[r, c] = i + 1

    plt.figure()
    plt.imshow(grid, cmap="Blues")
    plt.title("Trayectoria del agente")
    plt.colorbar()
    plt.savefig(tp)
    plt.close()

    return {
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_final": epsilon,
        "avg_reward_last_50": float(np.mean(rewards[-50:])),
        "demo_reward": demo_reward,
        "rewards_image": rewards_img,
        "trajectory_image": traj_img,
        "model_path": model_path,
    }
