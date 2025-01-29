# Dataset
DATASET_NAME = "cifar10"
DATA_DIR = "../../data"
CHECKPOINT_DIR = "checkpoints"
NUM_WORKERS = 0

# Training Hyperparameters
NUM_CHANNELS = 512
NUM_HIDDENS = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 5

# Model
NUM_LAYERS = 7

# Compute
ACCELERATOR = "gpu"
DEVICES = [0, 1]
PRECISION = 32

# Image
IMAGE_SIZE = 7 if DATASET_NAME == "mnist" else 8
NUM_SAMPLES = 64
NUM_CLASSES = 10

