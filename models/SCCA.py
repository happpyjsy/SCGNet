import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block

class SCCAModule(nn.Module):
    """Spatial Autocorrelation Token Analysis (SATA) Module"""
    def __init__(self, dim, gamma=0.7, alpha=1.0):
        super(SCCAModule, self).__init__()
        self.dim = dim  # Token embedding dimension
        self.gamma = gamma  # Threshold for applying SATA (e.g., 0.7 means apply to last 30% layers)
        self.alpha = alpha  # Controlling factor for bounds

    def compute_spatial_autocorrelation(self, x, attn):
        """Compute spatial autocorrelation scores (s) for tokens"""
        B, N, D = x.shape  # Batch size, number of tokens, dimension
        
        # Compute token-wise global context attribute 'a' (Eq. 6)
        a = x.mean(dim=-1)  # [B, N]
        
        # Normalize to get 'z' (Eq. 4)
        mu_a = a.mean(dim=1, keepdim=True)
        sigma_a = a.std(dim=1, keepdim=True) + 1e-6  # Avoid division by zero
        z = (a - mu_a) / sigma_a  # [B, N]
        
        # Use attention map as spatial weight matrix W (Eq. 3)
        W = attn  # [B, N, N]
        
        # Compute local Moran's I (Eq. 3)
        zzT = z.unsqueeze(-1) @ z.unsqueeze(-2)  # [B, N, N]
        I_l = torch.diagonal(zzT * W, dim1=-2, dim2=-1)  # [B, N]
        
        # Normalize to get 's' (Eq. 5)
        mu_I = I_l.mean(dim=1, keepdim=True)
        sigma_I = I_l.std(dim=1, keepdim=True) + 1e-6
        s = (I_l - mu_I) / sigma_I  # [B, N]
        
        return s

    def bipartite_matching(self, x, mask):
        """Bipartite Matching for token grouping"""
        B, N, D = x.shape
        indices = torch.nonzero(mask, as_tuple=False)  # Get indices of tokens in set A
        if indices.size(0) == 0:
            return x, torch.zeros(B, 0, D, device=x.device)
        
        tokens_A = x[indices[:, 0], indices[:, 1]]  # Extract tokens in set A
        N_A = tokens_A.size(0)
        if N_A <= 1:
            return x, tokens_A.unsqueeze(0)

        # Split into A1 and A2
        mid = N_A // 2
        A1, A2 = tokens_A[:mid], tokens_A[mid:]
        
        # Compute similarity (dot product)
        sim = A1 @ A2.T  # [mid, N_A - mid]
        matches = sim.argmax(dim=1)  # Find most similar token in A2 for each token in A1
        
        # Merge matched tokens
        merged_tokens = (A1 + A2[matches]) / 2  # Average features
        residual_mask = torch.ones(N_A - mid, dtype=torch.bool, device=x.device)
        residual_mask[matches] = False
        residual_tokens = A2[residual_mask]  # Unmatched tokens
        
        return merged_tokens, residual_tokens

    def forward(self, x, attn, layer_idx, total_layers):
        """Forward pass of SATA"""
        B, N, D = x.shape
        
        # Only apply SATA to layers >= gamma * total_layers
        if layer_idx < self.gamma * total_layers:
            return x
        
        # Compute spatial autocorrelation scores
        s = self.compute_spatial_autocorrelation(x, attn)
        
        # Compute bounds
        mu_s = s.mean(dim=1, keepdim=True)  # [B, 1]
        abs_median_s = torch.median(s.abs(), dim=1)[0].unsqueeze(1)  # [B, 1]
        lower_bound = self.alpha * (mu_s - abs_median_s)
        upper_bound = self.alpha * (mu_s + abs_median_s)
        
        # Token splitting (Eq. 8)
        mask_A = (s < lower_bound) | (s > upper_bound)  # [B, N]
        mask_B = ~mask_A  # [B, N]
        
        # Extract tokens for set B
        tokens_B = x * mask_B.unsqueeze(-1)  # [B, N, D], zero out tokens in A
        
        # Bipartite matching for set A
        merged_tokens, residual_tokens = self.bipartite_matching(x, mask_A)
        
        # Concatenate tokens_B (non-zero elements) and merged_tokens
        tokens_B_flat = tokens_B[mask_B].view(B, -1, D)  # [B, N_B, D]
        if merged_tokens.size(0) > 0:
            output_tokens = torch.cat([tokens_B_flat, merged_tokens], dim=1)  # [B, N_out, D]
        else:
            output_tokens = tokens_B_flat
        
        # Pad output_tokens to match FFN input requirement (optional)
        if output_tokens.size(1) < N:
            padding = torch.zeros(B, N - output_tokens.size(1), D, device=x.device)
            output_tokens = torch.cat([output_tokens, padding], dim=1)
        
        # Residual tokens to be concatenated after FFN
        return output_tokens, residual_tokens