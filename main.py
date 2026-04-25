from dataset import CelebADataset50K, build, get_test_image
import torch
from torch.utils.data import DataLoader
from utils import show
from model import UNet

def main():
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = torch.cuda.is_available()
   
    # dataset = CelebADataset50K(limit=BATCH_SIZE)
    # train_data, test_data = build(dataset)
   
    # train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    # test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    img = get_test_image()
    model = UNet()
    img_ = model.forward(img)
    print(img_.shape)


if __name__ == "__main__":
    main()