import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=t.device)) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8):
        super().__init__()
        self.block1 = nn.Sequential(nn.GroupNorm(groups, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.block2 = nn.Sequential(nn.GroupNorm(groups, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        h = h + self.time_proj(t)[:, :, None, None]
        return self.block2(h) + self.skip(x)

class ResPair(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block1 = ResidualBlock(in_ch, out_ch, time_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_dim)

    def forward(self, x, t):
        x = self.block1(x, t)
        return self.block2(x, t)

class UNet(nn.Module):
    def __init__(self, img_ch=3, channels=(64, 128, 256), time_dim=128):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.initial = nn.Conv2d(img_ch, channels[0], 3, padding=1)

        self.downs = nn.ModuleList([ResPair(ch, ch, time_dim) for ch in channels])
        self.downsamples = nn.ModuleList([nn.Conv2d(a, b, 4, 2, 1) for a, b in zip(channels, channels[1:])])

        self.middle = ResPair(channels[-1], channels[-1], time_dim)

        rev = list(zip(channels[:0:-1], channels[-2::-1]))
        self.upsamples = nn.ModuleList([nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(a, b, 3, padding=1)) for a, b in rev])
        self.ups = nn.ModuleList([ResPair(b * 2, b, time_dim) for _, b in rev])

        self.final = nn.Sequential(nn.GroupNorm(8, channels[0]), nn.SiLU(), nn.Conv2d(channels[0], img_ch, 3, padding=1))

    def forward(self, x, t):
        t = self.time_emb(t)
        x = self.initial(x)

        skips = []
        for i, down in enumerate(self.downs):
            x = down(x, t)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        x = self.middle(x, t)

        skips.pop()
        for upsample, up in zip(self.upsamples, self.ups):
            x = upsample(x)
            x = up(torch.cat([x, skips.pop()], dim=1), t)

        return self.final(x)