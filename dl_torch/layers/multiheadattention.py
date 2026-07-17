import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim:int, d_model:int, n_heads:int):
        super().__init__()
        self.W_Q = nn.Linear(input_dim, d_model, bias=False)
        self.W_K = nn.Linear(input_dim, d_model, bias=False)
        self.W_V = nn.Linear(input_dim, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = self.d_model // n_heads
        
        
    def forward(self, input_data:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        """
        the shape of input_data should be (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = input_data.shape
        Q = self.W_Q(input_data).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_K(input_data).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_V(input_data).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        
        S = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)
        
        if mask is not None:
            S = S.masked_fill(mask == 0, -1e9)
            
        A = F.softmax(S, dim=-1)
        
        O = A @ V
        O = O.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_O(O)