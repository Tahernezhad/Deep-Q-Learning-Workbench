import time
from pathlib import Path
import torch
import gymnasium as gym
from dqn_agent import DQNAgent


# 1. PATH TO YOUR RESULTS FOLDER
#    Copy the full path of a timestamped results folder here.
RESULTS_PATH = "results/LunarLander-v3_20250915_003640"

# 2. NUMBER OF EPISODES TO REPLAY
NUM_EPISODES = 5

# 3. SAVE A VIDEO OF THE REPLAY
SAVE_VIDEO = True

# 4. NAME FOR THIS RUN (Optional)
RUN_NAME = "IDE_Replay"


def main(results_path: str, num_episodes: int, run_name: str):
    """
    Loads a trained agent and runs it in the environment.
    """
    results_dir = Path(results_path)
    if not results_dir.is_dir() or "YOUR_RESULTS_FOLDER_HERE" in results_path:
        print(f"Error: Directory not found at '{results_dir}'")
        print("Please update the 'RESULTS_PATH' variable in this script with a valid path.")
        return

    # --- Load Configuration and Checkpoint ---
    config_path = results_dir / "hyperparameters.txt"
    checkpoint_path = results_dir / "best_model.pth"

    if not config_path.is_file() or not checkpoint_path.is_file():
        print(f"Error: Missing 'hyperparameters.txt' or 'best_model.pth' in {results_dir}")
        return

    class SimpleConfig:
        """A simple class to parse and hold hyperparameters from the saved text file."""

        def __init__(self, filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        if value.lower() == 'true':
                            setattr(self, key, True)
                        elif value.lower() == 'false':
                            setattr(self, key, False)
                        elif value.isdigit():
                            setattr(self, key, int(value))
                        elif '.' in value and all(part.isdigit() for part in value.split('.', 1)):
                            setattr(self, key, float(value))
                        else:
                            setattr(self, key, value)
            self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = SimpleConfig(config_path)

    print(f"--- Starting Replay Run: {run_name} ---")
    print(f"Loading model from: {checkpoint_path}")
    print(f"Environment: {config.ENV_NAME}, Model: {config.MODEL_TYPE}")
    print("-----------------------------------------")

    # --- Create Environment and Agent ---
    render_mode = "rgb_array" if SAVE_VIDEO else "human"
    env = gym.make(config.ENV_NAME, render_mode=render_mode)

    # Wrap the environment for video recording if enabled
    if SAVE_VIDEO:
        video_dir = results_dir / "replay_videos"
        video_dir.mkdir(exist_ok=True)
        print(f"Saving replay videos to: {video_dir}")
        env = gym.wrappers.RecordVideo(
            env,
            str(video_dir),
            episode_trigger=lambda x: True  # Record every episode
        )

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(n_states, n_actions, config)

    # --- Load Weights ---
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.policy_net.eval()

    # --- Run Replay Loop ---
    for i_episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, epsilon=0.01)  # Use learned policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            if render_mode == "human":
                time.sleep(0.02)  # Only slow down if rendering live

        print(f"Episode {i_episode + 1}/{num_episodes} | Reward: {episode_reward:.2f}")

    env.close()
    print("\nReplay finished.")


if __name__ == "__main__":
    # Call the main function with the configured variables
    main(RESULTS_PATH, NUM_EPISODES, RUN_NAME)