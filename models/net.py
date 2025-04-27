import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block
from patchhance import *
from SCCA import *

# 定义 SPM 模块
class SPM(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, kernel_sizes=[3, 5, 7, 9], window_size=2):
        super(SPM, self).__init__()
        self.csc = CrossScaleCom(in_channels, out_channels, num_heads, kernel_sizes)
        self.lfg = LocalFeatureGuided(out_channels, window_size)        # 局部增强

    def forward(self, x):
        x = self.msa(x)  # 聚合多尺度特征
        x = self.gle(x)  # 优化局部特征并减小分辨率
        return x

# 定义层次化的 ViT
# class SCGNET(VisionTransformer):
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12):
#         super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads)

#         # 定义 SPM 模块
#         self.spm1 = SPM(embed_dim, embed_dim * 2)  # 第一次合并
#         self.spm2 = SPM(embed_dim * 2, embed_dim * 4)  # 第二次合并

#         # 分阶段调整 Transformer 块
#         self.blocks = nn.ModuleList([
#             *[Block(embed_dim, num_heads) for _ in range(depth // 3)],         # 阶段 1
#             *[Block(embed_dim * 2, num_heads) for _ in range(depth // 3)],    # 阶段 2
#             *[Block(embed_dim * 4, num_heads) for _ in range(depth // 3)]     # 阶段 3
#         ])

#         # 调整分类头
#         self.head = nn.Linear(embed_dim * 4, num_classes)

#     def forward_features(self, x):
#         x = self.patch_embed(x)  # (batch_size, 196, 768)
#         x = self.pos_drop(x + self.pos_embed)

#         # 分阶段处理
#         for i, block in enumerate(self.blocks):
#             if i == len(self.blocks) // 3:  # 在第 4 层后插入 SPM1
#                 x = self.spm1(x)  # (batch_size, 49, 1536)
#             elif i == 2 * len(self.blocks) // 3:  # 在第 8 层后插入 SPM2
#                 x = self.spm2(x)  # (batch_size, 12, 3072)
#             x = block(x)

#         x = self.norm(x)
#         return x.mean(dim=1)  # 全局平均池化，(batch_size, 3072)

#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x


class SCGNET(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 gamma=0.7, alpha=1.0):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                         num_classes=num_classes, embed_dim=embed_dim, depth=depth, 
                         num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)

        # 定义 SPM 模块
        self.spm1 = SPM(embed_dim, embed_dim * 2)  # 第一次合并
        self.spm2 = SPM(embed_dim * 2, embed_dim * 4)  # 第二次合并

        # 定义 SATA 模块
        self.sata = SCCAModule(embed_dim, gamma=gamma, alpha=alpha)
        self.sata_mid = SCCAModule(embed_dim * 2, gamma=gamma, alpha=alpha)
        self.sata_last = SCCAModule(embed_dim * 4, gamma=gamma, alpha=alpha)

        # 分阶段调整 Transformer 块
        self.blocks = nn.ModuleList([
            *[Block(embed_dim, num_heads, mlp_ratio, qkv_bias) for _ in range(depth // 3)],      # 阶段 1
            *[Block(embed_dim * 2, num_heads, mlp_ratio, qkv_bias) for _ in range(depth // 3)],  # 阶段 2
            *[Block(embed_dim * 4, num_heads, mlp_ratio, qkv_bias) for _ in range(depth // 3)]   # 阶段 3
        ])

        # 调整分类头
        self.head = nn.Linear(embed_dim * 4, num_classes)

    def forward_features(self, x):
        x = self.patch_embed(x)  # [B, 196, 768]
        x = self.pos_drop(x + self.pos_embed)

        # 分阶段处理
        for i, block in enumerate(self.blocks):
            # Apply MHSA
            attn = block.attn(block.norm1(x))
            x = x + block.drop_path(attn)

            # Apply SATA
            if i < len(self.blocks) // 3:
                x_sata, residual = self.sata(x, attn, i, len(self.blocks))
            elif i < 2 * len(self.blocks) // 3:
                x_sata, residual = self.sata_mid(x, attn, i, len(self.blocks))
            else:
                x_sata, residual = self.sata_last(x, attn, i, len(self.blocks))

            # Apply FFN
            x_ffn = block.ffn(block.norm2(x_sata))
            x = x_sata + block.drop_path(x_ffn)

            # Concatenate residual tokens
            if residual.size(1) > 0:
                if x.size(1) > residual.size(1):
                    padding = torch.zeros_like(x[:, :x.size(1) - residual.size(1)])
                    residual = torch.cat([residual, padding], dim=1)
                x = x + residual

            # Apply SPM at stage boundaries
            if i == len(self.blocks) // 3 - 1:  # 阶段 1 结束
                x = self.spm1(x)  # [B, 49, 1536]
            elif i == 2 * len(self.blocks) // 3 - 1:  # 阶段 2 结束
                x = self.spm2(x)  # [B, 12, 3072]

        x = self.norm(x)
        return x.mean(dim=1)  # 全局平均池化，[B, 3072]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
# model = SCGNET()
# input = torch.randn(1, 3, 224, 224)
# output = model(input)
# print(output.shape)  # (1, 1000)