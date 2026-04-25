import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "./data/celeba-50K"
IMG_SIZE = 64
# BATCH_SIZE = 128
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = DEVICE == "cuda"
SPLIT_RATIO = 0.9
SEED = 42

LR = 2e-4
# EPOCHS = 100
EPOCHS = 1
T = 1000
BETA_START = 1e-4
BETA_END = 2e-2

IMG_CH = 3
# CHANNELS = (64, 128, 256)
CHANNELS = (32, 64, 128)
TIME_DIM = 128

SAVE_PATH = "ddpm_unet.pt"