import numpy as np
from ..layers.linear_normalization import LinearNormalization
from ..layers.multi_attention import MultiHeadAttention
from ..layers.positionffn import PositionwiseFeedForward

class TransformerEncoderBlock():
    def __init__(
        self, 
        input_dim:int,
        d_k:int, 
        d_v:int,
        head_num:int,
        d_model:int,
        d_ff:int,
        eps:float=1e-5,
    ) -> None:
       self.input_dim = input_dim
       self.d_k = d_k
       self.d_v = d_v
       self.head_num = head_num
       self.d_model = d_model
       self.d_ff = d_ff
       self.eps = eps
       self._init_models()

    def _init_models(self) -> None:
        self.MHA = MultiHeadAttention(
            input_dim=self.input_dim, 
            d_model=self.d_model,
            d_k=self.d_k, 
            d_v=self.d_v,
            head_num=self.head_num
            )
        
        self.norm1 = LinearNormalization(d_model=self.d_model, eps=self.eps)
        
        self.FFN = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_ff)
        
        self.norm2 = LinearNormalization(d_model=self.d_model, eps=self.eps)
    
    def forward(self, input_array:np.ndarray) -> np.ndarray:
        
        #sub-layer 1: x + MHA(LayerNorm(x))
        atten_out = self.MHA.forward(self.norm1.forward(input_array))
        out_1 = input_array + atten_out
        
        #sub-layer 2: x_1 + FFN(LayerNorm(x_1))
        ffn_out = self.FFN.forward(self.norm2.forward(out_1))
        out_2 = out_1 + ffn_out
        return out_2
        
    def backward(self, grad:np.ndarray) -> np.ndarray:
        #sub-layer 2: FFN
        d_ffn_out = grad
        d_norm2_out = self.FFN.backward(d_ffn_out)
        d_out1_norm = self.norm2.backward(d_norm2_out)
        d_out1 = grad + d_out1_norm
        
        #sub-layer 1:MHA
        d_atten_out = d_out1
        d_norm1_out = self.MHA.backward(d_atten_out)
        d_x_norm = self.norm1.backward(d_norm1_out)
        d_x = d_out1 + d_x_norm
        
        return d_x
