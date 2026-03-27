from __future__ import annotations

from torch import nn

from .blocks import DownBlock, ResidualBlock, SqueezeExcitation, UpBlock


class HRUNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        channels = [base_channels * multiplier for multiplier in channel_multipliers]
        self.out_channels = channels[0]

        self.stem = ResidualBlock(in_channels, channels[0])
        self.downs = nn.ModuleList(
            DownBlock(in_c, out_c) for in_c, out_c in zip(channels[:-1], channels[1:])
        )
        self.bottleneck = ResidualBlock(channels[-1], channels[-1])
        self.attention = SqueezeExcitation(channels[-1]) if use_attention else nn.Identity()
        self.ups = nn.ModuleList(
            UpBlock(in_c, skip_c, out_c)
            for in_c, skip_c, out_c in zip(
                reversed(channels[1:]),
                reversed(channels[:-1]),
                reversed(channels[:-1]),
            )
        )

    def forward(self, x):
        skips = [self.stem(x)]
        x = skips[0]
        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.attention(self.bottleneck(skips[-1]))
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return x
