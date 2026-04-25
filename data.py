from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import config as cfg

default_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

class CelebADataset50K(Dataset):
    def __init__(self, root=cfg.DATA_ROOT, transform=None):
        self.transform = default_transform if transform is None else transform
        self.paths = sorted(Path(root).glob("*.jpg"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)

def build(dataset, split_ratio=cfg.SPLIT_RATIO, seed=cfg.SEED):
    generator = torch.Generator().manual_seed(seed)
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size], generator=generator)

dataset = CelebADataset50K()
train_dataset, test_dataset = build(dataset)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=cfg.PIN_MEMORY,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=cfg.PIN_MEMORY,
)