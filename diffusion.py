import torch

def make_betas(device):
    import config as cfg
    return torch.linspace(cfg.BETA_START, cfg.BETA_END, cfg.T, device=device)

def forward_sample(x0, betas, t, eps=None):
    eps = torch.randn_like(x0) if eps is None else eps
    alpha_bar = torch.cumprod(1 - betas, dim=0)
    alpha_bar_t = alpha_bar[t][:, None, None, None]
    return alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * eps

@torch.no_grad()
def reverse_sample(model, betas, device):
    import config as cfg
    model.eval()

    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    x = torch.randn(1, cfg.IMG_CH, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device)

    for t in reversed(range(cfg.T)):
        t_batch = torch.tensor([t], device=device, dtype=torch.long)
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]

        eps_pred = model(x, t_batch)
        mean = (1 / alpha_t.sqrt()) * (
            x - ((1 - alpha_t) / (1 - alpha_bar_t).sqrt()) * eps_pred
        )

        if t > 0:
            x = mean + beta_t.sqrt() * torch.randn_like(x)
        else:
            x = mean

    model.train()
    return x