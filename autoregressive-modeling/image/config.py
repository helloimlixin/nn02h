# Dataset
DATASET_NAME = "cifar10"
DATA_DIR = "../../data"
CHECKPOINT_DIR = "checkpoints"
NUM_WORKERS = 0

# Training Hyperparameters
NUM_CHANNELS = 256
NUM_HIDDENS = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 256
NUM_EPOCHS = 100

# Model
NUM_LAYERS = 7

# Compute
ACCELERATOR = "gpu"
DEVICES = [0, 1]
PRECISION = 32

# Image
IMAGE_SIZE = 28 if DATASET_NAME == "mnist" else 32
NUM_SAMPLES = 64
NUM_CLASSES = 10

