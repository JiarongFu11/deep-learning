import torch
import torch.nn as nn
from typing import Optional

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model:int,
        head_num:int,
        d_ff:int, 
        dropout: float = 0.1
     ) -> None:
        super().__init__()
        self.MHA = nn.MultiheadAttention(d_model, head_num, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Input x shape: (batch_size, seq_len, d_model)
        Output x shape: (batch_size, seq_len, d_model)
        """
        # Sub-layer 1: Pre-LN Attention
        norm1_x = self.norm1(x)
        atten_out, _ = self.MHA(
            query=norm1_x,
            key=norm1_x,
            value=norm1_x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        out_1 = x + self.dropout1(atten_out)
        
        # Sub-layer 2: Pre-LN FFN
        norm2_x = self.norm2(out_1)
        ffn_out = self.FFN(norm2_x)
        out_2 = out_1 + ffn_out
        
        return out_2
        