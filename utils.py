import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def denorm(x):
    return (x.clamp(-1, 1) + 1) / 2

def show(x, nrow=4):
    x = denorm(x.detach().cpu())
    grid = make_grid(x, nrow=nrow)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

def save_images(x, path, nrow=4):
    x = denorm(x.detach().cpu())
    grid = make_grid(x, nrow=nrow)
    plt.imsave(path, grid.permute(1, 2, 0).numpy())