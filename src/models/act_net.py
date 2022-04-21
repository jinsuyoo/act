import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .common import (
    PreNorm,
    PreNorm2,
    ResBlock,
    Upsampler,
    MeanShift,
    FeedForward,
    default_conv,
    SelfAttention,
    CrossAttention,
)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self,
                 conv,
                 n_feat,
                 kernel_size,
                 reduction,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(conv,
                 n_feat,
                 kernel_size,
                 reduction,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1)
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class FB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=False, act=nn.ReLU(True)):
        super(FB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0:
                modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ACT(nn.Module):
    def __init__(self, args):

        super(ACT, self).__init__()

        conv = default_conv

        task = args.task
        scale = args.scale
        rgb_range = args.rgb_range
        n_colors = args.n_colors
        self.n_feats = n_feats = args.n_feats
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        reduction = args.reduction
        n_heads = args.n_heads
        n_layers = args.n_layers
        dropout_rate = args.dropout_rate
        n_fusionblocks = args.n_fusionblocks

        self.token_size = args.token_size
        self.n_fusionblocks = args.n_fusionblocks
        self.embedding_dim = embedding_dim = args.n_feats * (args.token_size**2)

        flatten_dim = embedding_dim
        hidden_dim = embedding_dim * args.expansion_ratio
        dim_head = embedding_dim // args.n_heads

        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # head module includes two residual blocks
        self.head = nn.Sequential(
            conv(args.n_colors, n_feats, 3),
            ResBlock(conv, n_feats, 5, act=nn.ReLU(True)),
            ResBlock(conv, n_feats, 5, act=nn.ReLU(True)),
        )

        # linear encoding after tokenization
        self.linear_encoding = nn.Linear(flatten_dim, embedding_dim)

        # conventional self-attention block inside Transformer Block
        self.mhsa_block = nn.ModuleList([
            nn.ModuleList([
                PreNorm(
                    embedding_dim, 
                    SelfAttention(embedding_dim, n_heads, dim_head, dropout_rate)
                ),
                PreNorm(
                    embedding_dim, 
                    FeedForward(embedding_dim, hidden_dim, dropout_rate)
                ),
            ]) for _ in range(n_layers // 2)
        ])

        # cross-scale token attention block inside Transformer Block
        self.csta_block = nn.ModuleList([
            nn.ModuleList([
                # FFN for large tokens before the cross-attention
                nn.Sequential(
                    nn.LayerNorm(embedding_dim * 2),
                    nn.Linear(embedding_dim * 2, embedding_dim // 2),
                    nn.GELU(),
                    nn.Linear(embedding_dim // 2, embedding_dim // 2)
                ),
                # Two cross-attentions
                PreNorm2(
                    embedding_dim // 2,
                    CrossAttention(embedding_dim // 2, n_heads // 2, dim_head, dropout_rate)
                ),
                PreNorm2(
                    embedding_dim // 2,
                    CrossAttention(embedding_dim // 2, n_heads // 2, dim_head, dropout_rate)
                ),
                # FFN for large tokens after the cross-attention
                nn.Sequential(
                    nn.LayerNorm(embedding_dim // 2),
                    nn.Linear(embedding_dim // 2, embedding_dim // 2),
                    nn.GELU(),
                    nn.Linear(embedding_dim // 2, embedding_dim * 2)
                ),
                # conventional FFN after the attention
                PreNorm(
                    embedding_dim,
                    FeedForward(embedding_dim, hidden_dim, dropout_rate)
                )
            ]) for _ in range(n_layers // 2)
        ])

        # CNN Branch borrowed from RCAN
        modules_body = [
            ResidualGroup(conv, n_feats, 3, reduction, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)
        ]

        modules_body.append(conv(n_feats, n_feats, 3))
        self.cnn_branch = nn.Sequential(*modules_body)

        # Fusion Blocks
        self.fusion_block = nn.ModuleList([
            nn.Sequential(
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True)),
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True)),
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True)),
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True)),
            ) for _ in range(n_fusionblocks)
        ])

        self.fusion_mlp = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embedding_dim),
            ) for _ in range(n_fusionblocks - 1)
        ])

        self.fusion_cnn = nn.ModuleList([
            nn.Sequential(
                conv(n_feats, n_feats, 3), nn.ReLU(True), conv(n_feats, n_feats, 3)
            ) for _ in range(n_fusionblocks - 1)
        ])

        # single convolution to lessen dimension after body module
        self.conv_last = conv(n_feats * 2, n_feats, 3)

        # tail module
        if task == 'sr':
            self.tail = nn.Sequential(
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, n_colors, 3),
            )
        elif task == 'car':
            self.tail = conv(n_feats, n_colors, 3)

    def forward(self, x):
        h, w = x.shape[-2:]

        x = self.sub_mean(x)
        x = self.head(x)

        identity = x

        x_tkn = F.unfold(x, self.token_size, stride=self.token_size)
        x_tkn = rearrange(x_tkn, 'b d t -> b t d')

        x_tkn = self.linear_encoding(x_tkn) + x_tkn

        for i in range(self.n_fusionblocks):
            x_tkn = self.mhsa_block[i][0](x_tkn) + x_tkn
            x_tkn = self.mhsa_block[i][1](x_tkn) + x_tkn

            x_tkn_a, x_tkn_b = torch.split(x_tkn, self.embedding_dim // 2, -1)

            x_tkn_b = rearrange(x_tkn_b, 'b t d -> b d t')
            x_tkn_b = F.fold(x_tkn_b, (h, w), self.token_size, stride=self.token_size)

            x_tkn_b = F.unfold(x_tkn_b, self.token_size * 2, stride=self.token_size)
            x_tkn_b = rearrange(x_tkn_b, 'b d t -> b t d')

            x_tkn_b = self.csta_block[i][0](x_tkn_b)
            _x_tkn_a, _x_tkn_b = x_tkn_a, x_tkn_b
            x_tkn_a = self.csta_block[i][1](x_tkn_a, _x_tkn_b) + x_tkn_a
            x_tkn_b = self.csta_block[i][2](x_tkn_b, _x_tkn_a) + x_tkn_b
            x_tkn_b = self.csta_block[i][3](x_tkn_b)

            x_tkn_b = rearrange(x_tkn_b, 'b t d -> b d t')
            x_tkn_b = F.fold(x_tkn_b, (h, w), self.token_size * 2, stride=self.token_size)

            x_tkn_b = F.unfold(x_tkn_b, self.token_size, stride=self.token_size)
            x_tkn_b = rearrange(x_tkn_b, 'b d t -> b t d')

            x_tkn = torch.cat((x_tkn_a, x_tkn_b), -1)
            x_tkn = self.csta_block[i][4](x_tkn) + x_tkn

            x = self.cnn_branch[i](x)

            x_tkn_res, x_res = x_tkn, x

            x_tkn = rearrange(x_tkn, 'b t d -> b d t')
            x_tkn = F.fold(x_tkn, (h, w), self.token_size, stride=self.token_size)

            f = torch.cat((x, x_tkn), 1)
            f = f + self.fusion_block[i](f)

            if i != (self.n_fusionblocks - 1):
                x_tkn, x = torch.split(f, self.n_feats, 1)

                x_tkn = F.unfold(x_tkn, self.token_size, stride=self.token_size)
                x_tkn = rearrange(x_tkn, 'b d t -> b t d')
                x_tkn = self.fusion_mlp[i](x_tkn)+ x_tkn_res

                x = self.fusion_cnn[i](x) + x_res

        x = self.conv_last(f)

        x = x + identity

        x = self.tail(x)
        x = self.add_mean(x)

        return x
