import random
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across different libraries.
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_rewards(rewards: list[float], save_path: Path, moving_avg_window: int):
    """
    Plots the total rewards per episode and saves the figure.
    Args:
        rewards (list[float]): A list of total rewards for each episode.
        save_path (Path): The path to save the plot image.
        moving_avg_window (int): The window size for the moving average.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward')
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    if len(rewards) >= moving_avg_window:
        moving_avg = np.convolve(rewards, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
        plt.plot(np.arange(moving_avg_window - 1, len(rewards)), moving_avg,
                 label=f'{moving_avg_window}-episode MA')

    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")


def save_checkpoint(agent, save_path: Path):
    """
    Saves the model's state dictionary.
    Args:
        agent: The agent whose policy network should be saved.
        save_path (Path): The path to save the model checkpoint.
    """
    checkpoint = {'policy_net_state_dict': agent.policy_net.state_dict()}
    torch.save(checkpoint, save_path)



def save_config(config_module, save_path: Path):
    """
    Saves the hyperparameter configuration to a text file.
    Args:
        config_module: The config module itself.
        save_path (Path): Path to save the configuration file.
    """
    with open(save_path, 'w') as f:
        f.write("HYPERPARAMETER CONFIGURATION\n")
        f.write("=" * 30 + "\n")
        params = {k: v for k, v in config_module.__dict__.items() if k.isupper()}
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"Configuration saved to {save_path}")
