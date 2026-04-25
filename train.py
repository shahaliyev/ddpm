import os
import torch
import torch.nn as nn
import config as cfg
from model import UNet
from data import train_loader
from diffusion import make_betas, forward_sample, reverse_sample
from torchvision.utils import save_image
from tqdm import tqdm


def main():
    os.makedirs(cfg.RUNS_DIR, exist_ok=True)

    device = cfg.DEVICE
    betas = make_betas(device)

    model = UNet(
        img_ch=cfg.IMG_CH,
        channels=cfg.CHANNELS,
        time_dim=cfg.TIME_DIM
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.LR)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=cfg.EPOCHS,
        eta_min=cfg.LR * 0.05
    )

    loss_fn = nn.MSELoss()

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0

    if os.path.exists(cfg.SAVE_PATH):
        ckpt = torch.load(cfg.SAVE_PATH, map_location=device)

        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt["epoch"] + 1

        print(f"Resuming from epoch {start_epoch + 1}")

    for e in range(start_epoch, cfg.EPOCHS):
        model.train()
        total_loss = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {e + 1}/{cfg.EPOCHS}")

        for X in pbar:
            X = X.to(device, non_blocking=cfg.PIN_MEMORY)
            B = X.shape[0]

            t = torch.randint(0, cfg.T, (B,), device=device)
            eps = torch.randn_like(X)
            X_t = forward_sample(X, betas, t, eps)

            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                eps_pred = model(X_t, t)
                loss = loss_fn(eps_pred, eps)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            total_loss += loss.item() * B
            n += B

            pbar.set_postfix(
                loss=loss.item(),
                avg_loss=total_loss / n,
                lr=scheduler.get_last_lr()[0]
            )

        avg_loss = total_loss / n
        scheduler.step()

        print(
            f"Epoch {e + 1}: "
            f"Loss {avg_loss:.6f}, "
            f"LR {scheduler.get_last_lr()[0]:.8f}"
        )

        torch.save(
            {
                "epoch": e,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "loss": avg_loss,
                "config": {
                    "img_ch": cfg.IMG_CH,
                    "channels": cfg.CHANNELS,
                    "time_dim": cfg.TIME_DIM,
                    "lr": cfg.LR,
                    "epochs": cfg.EPOCHS,
                    "T": cfg.T,
                    "beta_start": cfg.BETA_START,
                    "beta_end": cfg.BETA_END,
                },
            },
            cfg.SAVE_PATH
        )

        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                sample = reverse_sample(model, betas, device)

            sample = ((sample + 1) / 2).clamp(0, 1)
            save_image(sample, os.path.join(cfg.RUNS_DIR, f"epoch_{e + 1:04d}.png"))


if __name__ == "__main__":
    main()