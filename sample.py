import os
import torch
from torchvision.utils import save_image

from model import UNet
from config import Config


@torch.no_grad()
def sample(model, betas, img_size, channels, n_samples, device):
    model.eval()

    T = len(betas)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    x = torch.randn(n_samples, channels, img_size, img_size, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]

        eps_pred = model(x, t_batch)

        mean = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred
        )

        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean

    return x


def main():
    cfg = Config()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet().to(device)
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))

    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T, device=device)

    samples = sample(
        model=model,
        betas=betas,
        img_size=cfg.img_size,
        channels=cfg.channels,
        n_samples=cfg.n_samples,
        device=device,
    )

    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    os.makedirs(cfg.sample_dir, exist_ok=True)
    save_image(samples, os.path.join(cfg.sample_dir, "generated.png"), nrow=8)


if __name__ == "__main__":
    main()