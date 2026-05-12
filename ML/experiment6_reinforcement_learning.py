# ============================================================
# Experiment No. 6 - Reinforcement Learning: Maze Exploration
# Subject: Machine Learning (417521) | BE AI&DS
# Aim: Train an agent to navigate a maze using Q-Learning
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# SECTION 1: MAZE ENVIRONMENT
# ══════════════════════════════════════════════════════════════

class MazeEnvironment:
    """
    Maze represented as a 2D grid.
    0 = free path, 1 = wall, S = start, G = goal
    """
    def __init__(self, maze=None):
        if maze is None:
            # Default 8x8 maze (0=path, 1=wall)
            self.maze = np.array([
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0],
                [0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0],
            ])
        else:
            self.maze = np.array(maze)

        self.rows, self.cols = self.maze.shape
        self.start = (0, 0)
        self.goal  = (self.rows - 1, self.cols - 1)

        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_names = {0: '↑ Up', 1: '↓ Down', 2: '← Left', 3: '→ Right'}
        self.n_actions = 4
        self.n_states  = self.rows * self.cols

        self.reset()

    def reset(self):
        """Reset agent to start position."""
        self.agent_pos = list(self.start)
        return self._state()

    def _state(self):
        """Convert (row, col) to a single integer state."""
        return self.agent_pos[0] * self.cols + self.agent_pos[1]

    def step(self, action):
        """
        Execute action; return (next_state, reward, done).
        Rewards:
          +100 for reaching the goal
          -10  for hitting a wall
          -1   for each valid step (encourages shortest path)
        """
        dr, dc = self.actions[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        # Check boundaries
        if new_r < 0 or new_r >= self.rows or new_c < 0 or new_c >= self.cols:
            return self._state(), -10, False      # Out of bounds

        # Check wall
        if self.maze[new_r, new_c] == 1:
            return self._state(), -10, False      # Hit a wall

        # Valid move
        self.agent_pos = [new_r, new_c]

        if self.agent_pos == list(self.goal):
            return self._state(), +100, True      # Goal reached!

        return self._state(), -1, False            # Normal step

    def render(self, path=None, title="Maze"):
        """Visualize the maze, optionally with agent path."""
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['white', 'black']
        cmap = LinearSegmentedColormap.from_list('maze', colors, N=2)

        ax.imshow(self.maze, cmap=cmap, vmin=0, vmax=1)

        # Grid lines
        for x in range(self.cols + 1):
            ax.axvline(x - 0.5, color='gray', linewidth=0.5)
        for y in range(self.rows + 1):
            ax.axhline(y - 0.5, color='gray', linewidth=0.5)

        # Draw path
        if path:
            path_rows = [p[0] for p in path]
            path_cols = [p[1] for p in path]
            ax.plot(path_cols, path_rows, 'b-o', linewidth=2.5,
                    markersize=6, label='Agent Path', zorder=3)

        # Start and Goal markers
        ax.text(self.start[1], self.start[0], 'S', ha='center', va='center',
                fontsize=18, fontweight='bold', color='green', zorder=4)
        ax.text(self.goal[1], self.goal[0], 'G', ha='center', va='center',
                fontsize=18, fontweight='bold', color='red', zorder=4)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        if path:
            ax.legend(loc='upper right')
        plt.tight_layout()
        return fig

# ══════════════════════════════════════════════════════════════
# SECTION 2: Q-LEARNING AGENT
# ══════════════════════════════════════════════════════════════

class QLearningAgent:
    """
    Q-Learning: Off-policy temporal difference algorithm.

    Q-Table update rule:
    Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') − Q(s, a)]

    Where:
      α (alpha)  = learning rate
      γ (gamma)  = discount factor
      ε (epsilon) = exploration rate (epsilon-greedy policy)
    """
    def __init__(self, n_states, n_actions,
                 alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha          # Learning rate
        self.gamma     = gamma          # Discount factor
        self.epsilon   = epsilon        # Initial exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        # Initialize Q-table with zeros
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        """Epsilon-greedy policy: explore or exploit."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)   # Explore
        return np.argmax(self.Q[state])                 # Exploit

    def learn(self, state, action, reward, next_state, done):
        """Update Q-table using Bellman equation."""
        q_current = self.Q[state, action]
        q_target  = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += self.alpha * (q_target - q_current)

    def decay_epsilon(self):
        """Reduce exploration rate over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ══════════════════════════════════════════════════════════════
# SECTION 3: TRAINING LOOP
# ══════════════════════════════════════════════════════════════

print("=== Reinforcement Learning: Maze Q-Learning ===\n")

# Initialize environment and agent
env = MazeEnvironment()
agent = QLearningAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    alpha=0.1,
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01
)

# Visualize the maze
fig_maze = env.render(title="Maze Environment  (S=Start, G=Goal, Black=Wall)")
plt.savefig('maze.png', dpi=150)
plt.show()

# Hyperparameters
N_EPISODES = 1000
MAX_STEPS  = 200

# Training metrics
episode_rewards   = []
episode_steps     = []
success_history   = []

print(f"Training for {N_EPISODES} episodes...")

for episode in range(N_EPISODES):
    state = env.reset()
    total_reward = 0
    success = False

    for step in range(MAX_STEPS):
        action     = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state        = next_state
        total_reward += reward

        if done:
            success = True
            break

    agent.decay_epsilon()
    episode_rewards.append(total_reward)
    episode_steps.append(step + 1)
    success_history.append(int(success))

    if (episode + 1) % 100 == 0:
        recent_success = sum(success_history[-100:])
        print(f"  Episode {episode+1:4d} | Avg Reward (last 100): "
              f"{np.mean(episode_rewards[-100:]):7.1f} | "
              f"Success rate: {recent_success}% | ε: {agent.epsilon:.3f}")

# ══════════════════════════════════════════════════════════════
# SECTION 4: PLOT TRAINING RESULTS
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Smoothed reward
window = 50
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
axes[0].plot(episode_rewards, alpha=0.3, color='steelblue', label='Raw')
axes[0].plot(range(window-1, len(episode_rewards)), smoothed,
             color='steelblue', linewidth=2, label=f'{window}-ep avg')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Total Reward')
axes[0].set_title('Training Rewards')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.4)

# Steps per episode
smoothed_steps = np.convolve(episode_steps, np.ones(window)/window, mode='valid')
axes[1].plot(episode_steps, alpha=0.3, color='coral')
axes[1].plot(range(window-1, len(episode_steps)), smoothed_steps,
             color='coral', linewidth=2)
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Steps to Goal')
axes[1].set_title('Steps per Episode')
axes[1].grid(True, linestyle='--', alpha=0.4)

# Success rate (rolling 100 episodes)
rolling_success = [np.mean(success_history[max(0,i-100):i+1])*100
                   for i in range(len(success_history))]
axes[2].plot(rolling_success, color='mediumseagreen', linewidth=2)
axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Success Rate (%)')
axes[2].set_title('Rolling Success Rate (100 ep window)')
axes[2].set_ylim(0, 105)
axes[2].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.4)

plt.suptitle('Q-Learning Training Progress', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print("Training curves saved.")

# ══════════════════════════════════════════════════════════════
# SECTION 5: EXTRACT OPTIMAL POLICY & TRACE PATH
# ══════════════════════════════════════════════════════════════

print("\n=== Extracting Optimal Policy ===")

# Derive optimal path by following greedy policy from start
def trace_optimal_path(env, agent, max_steps=100):
    state = env.reset()
    path  = [tuple(env.agent_pos)]

    for _ in range(max_steps):
        action = np.argmax(agent.Q[state])  # Pure greedy (no exploration)
        next_state, reward, done = env.step(action)
        path.append(tuple(env.agent_pos))
        state = next_state
        if done:
            print(f"  Goal reached in {len(path)-1} steps!")
            return path

    print("  Could not reach goal (maze may require more training).")
    return path

optimal_path = trace_optimal_path(env, agent)

# Visualize optimal path
env.reset()
fig_path = env.render(path=optimal_path,
                      title=f"Optimal Path Found by Q-Learning ({len(optimal_path)-1} steps)")
plt.savefig('optimal_path.png', dpi=150)
plt.show()
print("Optimal path visualization saved.")

# ══════════════════════════════════════════════════════════════
# SECTION 6: Q-TABLE HEATMAP
# ══════════════════════════════════════════════════════════════

q_max = np.max(agent.Q, axis=1).reshape(env.rows, env.cols)

plt.figure(figsize=(8, 6))
sns_mask = env.maze.astype(bool)   # Mask walls
plt.imshow(q_max, cmap='RdYlGn')

for i in range(env.rows):
    for j in range(env.cols):
        if env.maze[i, j] == 0:
            plt.text(j, i, f'{q_max[i,j]:.0f}', ha='center', va='center',
                     fontsize=7, color='black')
        else:
            plt.text(j, i, '■', ha='center', va='center', fontsize=12, color='white')

plt.colorbar(label='Max Q-Value')
plt.title('Q-Value Heatmap (Max Q per cell)')
plt.axis('off')
plt.tight_layout()
plt.savefig('q_heatmap.png', dpi=150)
plt.show()

# Print policy arrows
print("\n=== Optimal Policy (Action at each cell) ===")
arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
for i in range(env.rows):
    row_str = ""
    for j in range(env.cols):
        if env.maze[i, j] == 1:
            row_str += " ■ "
        elif (i, j) == env.start:
            row_str += " S "
        elif (i, j) == env.goal:
            row_str += " G "
        else:
            state  = i * env.cols + j
            a_best = np.argmax(agent.Q[state])
            row_str += f" {arrows[a_best]} "
    print(row_str)

# ──────────────────────────────────────────────
# Conclusion
# ──────────────────────────────────────────────
final_success = sum(success_history[-100:])
print(f"\n=== Conclusion ===")
print(f"Q-Learning successfully trained an agent to navigate an 8x8 maze.")
print(f"Final success rate (last 100 episodes): {final_success}%")
print(f"Optimal path length: {len(optimal_path)-1} steps")
print("Key parameters: α=0.1 (learning rate), γ=0.95 (discount), ε-greedy exploration.")
