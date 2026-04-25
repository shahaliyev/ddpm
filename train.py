import torch
import torch.nn as nn
import config as cfg
from model import UNet
from data import train_loader
from diffusion import make_betas, forward_sample

def main():
    device = cfg.DEVICE
    betas = make_betas(device)

    model = UNet(img_ch=cfg.IMG_CH, channels=cfg.CHANNELS, time_dim=cfg.TIME_DIM).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
    loss_fn = nn.MSELoss()

    for e in range(cfg.EPOCHS):
        model.train()
        total_loss = 0.0
        n = 0

        for X in train_loader:
            X = X.to(device, non_blocking=cfg.PIN_MEMORY)
            B = X.shape[0]

            t = torch.randint(0, cfg.T, (B,), device=device)
            eps = torch.randn_like(X)
            X_t = forward_sample(X, betas, t, eps)

            eps_pred = model(X_t, t)
            loss = loss_fn(eps_pred, eps)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * B
            n += B

        print(f"Epoch {e + 1}: Loss {total_loss / n:.6f}")
        torch.save(model.state_dict(), cfg.SAVE_PATH)

if __name__ == "__main__":
    main()