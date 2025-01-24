# Training Hyperparameters
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 100

# Dataset
DATA_DIR = "../data"
NUM_WORKERS = 1

# Compute
ACCELERATOR = "gpu"
DEVICES = [0, 1]
PRECISION = 32
