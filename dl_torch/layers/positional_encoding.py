import torch 
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, max_len:int, d_model:int, dropout:float=0.1):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout=dropout)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        even_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
        div_terms = torch.exp(-even_indices / d_model * math.log(10000.0)).unsqueeze(0)
        angle_rates = pos * div_terms
        
        self.pe = torch.zeros((self.max_len, self.d_model))
        self.pe[:, 0::2] = torch.sin(angle_rates)
        self.pe[:, 1::2] = torch.cos(angle_rates)
        
        self.pe = self.pe.unsqueeze(0)
        
        self.register_buffer('pe', self.pe)
        
    def forward(self, x:torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        
        assert seq_len <= self.max_len, f"Input seq_len {seq_len} exceeds max_len {self.max_len}"
        assert d_model == self.d_model, f"Dimension mismatch: expected {self.d_model}, got {d_model}"
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
        