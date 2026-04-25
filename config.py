import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "./data/celeba-50K"
IMG_SIZE = 64
BATCH_SIZE = 256
NUM_WORKERS = 4
PIN_MEMORY = DEVICE == "cuda"
SPLIT_RATIO = 0.9
SEED = 42

LR = 3e-4
EPOCHS = 100
T = 1000
BETA_START = 1e-4
BETA_END = 2e-2

IMG_CH = 3
CHANNELS = (64, 128, 256, 512)
TIME_DIM = 128

SAVE_PATH = "ddpm_unet.pt"
RUNS_DIR = "runs"