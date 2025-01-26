# Dataset
DATASET_NAME = "cifar10"
DATA_DIR = "../../data"
CHECKPOINT_DIR = "checkpoints"
NUM_WORKERS = 0

# Training Hyperparameters
NUM_CHANNELS = 1 if DATASET_NAME == "mnist" else 3
NUM_HIDDENS = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 100

# Compute
ACCELERATOR = "gpu"
DEVICES = [0, 1]
PRECISION = 32

# Image
IMAGE_SIZE = 28 if DATASET_NAME == "mnist" else 32

