import torch

def make_betas(device):
    import config as cfg
    return torch.linspace(cfg.BETA_START, cfg.BETA_END, cfg.T, device=device)

def forward_sample(x0, betas, t, eps=None):
    eps = torch.randn_like(x0) if eps is None else eps
    alpha_bar = torch.cumprod(1 - betas, dim=0)
    alpha_bar_t = alpha_bar[t][:, None, None, None]
    return alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * eps