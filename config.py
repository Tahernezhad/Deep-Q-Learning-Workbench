import torch


ENV_NAME = 'CartPole-v1'  # --- CHANGE ENVIRONMENT HERE ---
# Other popular environments: 'LunarLander-v2', 'Acrobot-v1', 'MountainCar-v0'

# --- Agent Hyperparameters ---
LEARNING_RATE = 0.0005       # Learning rate for the optimizer
GAMMA = 0.99                # Discount factor for future rewards
BATCH_SIZE = 64             # Number of samples per training batch
REPLAY_BUFFER_SIZE = 100000 # Maximum size of the replay buffer
TARGET_UPDATE_FREQ = 1000   # How often to update the target network (in steps)
HIDDEN_LAYER_SIZE = 256     # Number of neurons in hidden layers

# --- Epsilon-Greedy Strategy ---
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000  # Epsilon decays over this many steps

# --- DQN Variants ---
double_dqn = True
dueling_network = True

# --- Training Loop Settings ---
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000
REPORT_INTERVAL = 100
MOVING_AVG_WINDOW = 100
SEED = 42

# --- Device Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Save/Load Settings ---
SAVE_MODEL = True
RECORD_VIDEO = True
VIDEO_RECORD_INTERVAL = 50  # Record a video every N episodes

