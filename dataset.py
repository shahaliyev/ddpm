from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import random_split

def get_test_image(root='./data/celeba-50K', filename='000020.jpg', transform=transforms.ToTensor()):
    path = os.path.join(root, filename)
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(dim=0)


class CelebADataset50K(Dataset):
    def __init__(self, 
                 root='./data/celeba-50K', 
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),]
                 ), 
                 limit=50000):
        self.root = root
        self.transform = transform
        self.paths = sorted(Path(root).glob("*.jpg"))[:limit]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)



def build(dataset, split_ratio=0.9, seed=42):
    generator = torch.Generator().manual_seed(seed)
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size], 
        generator=generator
    )
    return train_dataset, test_dataset


