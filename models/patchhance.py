import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossScaleCom(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, kernel_sizes=[3, 5, 7, 9]):
        super(CrossScaleCom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel_sizes = kernel_sizes
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.channels_per_head = in_channels // num_heads
        
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(self.channels_per_head, self.channels_per_head, kernel_size=k, 
                      padding=k//2, groups=self.channels_per_head)
            for k in kernel_sizes
        ])
        
        # Channel shuffle to interleave features
        self.channel_shuffle = lambda x: x.view(x.size(0), self.num_heads, self.channels_per_head, 
                                                x.size(2), x.size(3)).transpose(1, 2).contiguous()
        
        # Linear projections to fuse multi-scale features
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.num_heads, self.channels_per_head, H, W)
        
        head_outputs = []
        for i, dw_conv in enumerate(self.dw_convs):
            head_outputs.append(dw_conv(x[:, i]))
        x = torch.stack(head_outputs, dim=1)  # [B, num_heads, channels_per_head, H, W]
        
        # Channel shuffle
        x = self.channel_shuffle(x)
        x = x.view(B, C, H, W)
        
        x = self.proj(x)
        return x

class LocalFeatureGuided(nn.Module):
    def __init__(self, in_channels, window_size=2, kernel_size=7):
        super(LocalFeatureGuided, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        
        # this is guide generator module(GGM)
        self.ggm = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                      stride=window_size, padding=kernel_size//2, groups=in_channels)
        )
        
        # Self-attention parameters
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Guide tokens
        guide_tokens = self.ggm(x)  # [B, C, H//window_size, W//window_size]
        
        # Unfold input into local windows
        x_unfold = F.unfold(x, kernel_size=self.window_size, stride=self.window_size)
        x_unfold = x_unfold.view(B, C, self.window_size * self.window_size, -1)  # [B, C, k^2, num_windows]
        num_windows = x_unfold.size(-1)
        
        # Prepare tokens with Guide tokens
        guide_tokens = guide_tokens.view(B, C, 1, num_windows)
        tokens = torch.cat([guide_tokens, x_unfold], dim=2)  # [B, C, k^2 + 1, num_windows]
        
        # Self-attention
        qkv = self.qkv(tokens)  # [B, 3C, k^2 + 1, num_windows]
        q, k, v = qkv.chunk(3, dim=1)  # Each: [B, C, k^2 + 1, num_windows]
        
        attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)  # [B, k^2 + 1, k^2 + 1, num_windows]
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, k^2 + 1, C, num_windows]
        
        # Extract Guide token output
        out = out[:, 0, :, :].view(B, C, H // self.window_size, W // self.window_size)  # [B, C, H', W']
        out = self.proj(out)
        return out

class StepwisePatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, kernel_sizes=[3, 5, 7, 9], window_size=2):
        super(StepwisePatchMerging, self).__init__()
        self.csc = CrossScaleCom(in_channels, out_channels, num_heads, kernel_sizes)
        self.lfg = LocalFeatureGuided(out_channels, window_size)

    def forward(self, x):
        x = self.csc(x)  
        x = self.lfg(x)  
        return x

if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    
    
    spm = StepwisePatchMerging(in_channels=64, out_channels=128)
    
    
    output = spm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")