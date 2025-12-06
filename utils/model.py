from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.bias_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads)
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor):
        B, N, E = x.shape

        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # B,H,N,N

        # if self.pairwise: # add pairwise bias only if enabled
        #     if pairwise_feats is None:
        #         raise ValueError("pairwise_feats must be provided when pairwise is True")
        #     bias_logits = self.bias_mlp(pairwise_feats)  # (B, N, N, H)
        #     bias_logits = bias_logits.permute(0, 3, 1, 2)  # (B, H, N, N)
        #     scores = scores + bias_logits
        
        # if key_padding_mask is not None:
        #     mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # B,1,1,N
        #     scores = scores.masked_fill(mask == True, float('-inf'))

        if pad_mask is not None:
            mask = pad_mask.unsqueeze(1).unsqueeze(2)  # B,1,1,N
            scores = scores.masked_fill(mask == True, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # B,H,N,N
        out = torch.matmul(attn, V)  # B,H,N,head_dim

        out = out.transpose(1, 2).contiguous().view(B, N, E)
        out = self.out_proj(out)
        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            num_heads: int, 
            dim_feedforward: int = 2048,
            dropout_rate: float = 0.1, 
        ):
        super().__init__()
        self.self_attn = AttentionLayer(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.activation = nn.ReLU()

    def forward(
            self, 
            src: torch.Tensor, 
            pad_mask: torch.Tensor
        ):
        src2 = self.self_attn(src, pad_mask)
        src = src + self.dropout1(src2) # Residual connection with dropout
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))) # Feedforward network with dropout
        src = src + self.dropout2(src2)
        src = self.norm2(src) # Final layer normalization
        return src

class TransformerEncoder(nn.Module):
    def __init__(
            self, 
            num_features: int, # Number of jet features
            embed_size: int,   # Embedding size
            num_heads: int = 8, 
            num_layers: int = 4,
        ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_size, 
                    num_heads, 
                ) for _ in range(num_layers)
            ]
        )
        self.norm_cls_embedding = nn.LayerNorm(embed_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor):
        B = x.size(0)
        x = self.input_proj(x) # (B, N, num_features) -> (B, num_jets, embed_size)
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, embed_size)
        x = torch.cat([cls_tokens, x], dim=1) # (B, num_jets+1, embed_size)

        pad_mask = torch.cat(
            [
                torch.zeros(pad_mask.size(0), 1, dtype=torch.bool, device=pad_mask.device),
                pad_mask
            ],
            dim=1
        )
        for layer in self.layers:
            x = layer(x, pad_mask)
        latent = self.norm_cls_embedding(x[:, 0, :])
        return latent

class MLPClassifier(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            hidden_dim: int,
            hidden_layers: int,
            output_dim: int = 1,
            dropout_rate: float = 0.1,
        ):
        super().__init__()
        # Add dropout
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(hidden_layers + 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # softmax
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = self.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x