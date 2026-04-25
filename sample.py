import torch
import config as cfg
from model import UNet
from diffusion import make_betas
from utils import show, save_images

@torch.no_grad()
def sample(model, betas, n=16):
    device = cfg.DEVICE
    x = torch.randn(n, cfg.IMG_CH, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device)

    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    for t in reversed(range(cfg.T)):
        ts = torch.full((n,), t, device=device, dtype=torch.long)
        eps_pred = model(x, ts)

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = betas[t]

        x = (x - ((1 - alpha_t) / (1 - alpha_bar_t).sqrt()) * eps_pred) / alpha_t.sqrt()

        if t > 0:
            x = x + beta_t.sqrt() * torch.randn_like(x)

    return x

def main():
    device = cfg.DEVICE
    betas = make_betas(device)

    model = UNet(img_ch=cfg.IMG_CH, channels=cfg.CHANNELS, time_dim=cfg.TIME_DIM).to(device)
    model.load_state_dict(torch.load(cfg.SAVE_PATH, map_location=device))
    model.eval()

    x = sample(model, betas, n=16)
    show(x)
    save_images(x, "samples.png")

if __name__ == "__main__":
    main()