# Training Hyperparameters
NUM_CHANNELS = 1
NUM_HIDDENS = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Dataset
DATA_DIR = "../../data"
CHECKPOINT_DIR = "../../checkpoints"
NUM_WORKERS = 0

# Compute
ACCELERATOR = "gpu"
DEVICES = [0, 1]
PRECISION = 32
