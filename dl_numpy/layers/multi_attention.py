import numpy as np

class MultiHeadAttention():
    def __init__(self, input_dim, d_model, d_k, d_v, head_num):
        self.input_dim = input_dim
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.head_num = head_num
        self._init_params()
    
    def _init_params(self,):
        assert self.d_model == self.head_num * self.d_k
        limit = np.sqrt(6 / (self.d_model + self.d_model))
        
        self.W_q = np.random.uniform(-limit, limit, size=(self.d_model, self.d_model))
        self.W_k = np.random.uniform(-limit, limit, size=(self.d_model, self.d_model))
        self.W_v = np.random.uniform(-limit, limit, size=(self.d_model, self.d_model))
        
        self.W_o = np.random.uniform(-limit, limit, size=(self.d_model, self.d_model))
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_data = input_data
        
        B, S, D = input_data.shape
        H = self.head_num
        d_k = self.d_k
        d_v = self.d_v
        
        self.Q_proj = input_data @ self.W_q
        self.K_proj = input_data @ self.W_k
        self.V_proj = input_data @ self.W_v
        
        Q_reshaped = self.Q_proj.reshape(B, S, H, d_k)
        K_reshaped = self.K_proj.reshape(B, S, H, d_k)
        V_reshaped = self.V_proj.reshape(B, S, H, d_v)
        
        self.Q = Q_reshaped.transpose(0, 2, 1, 3)
        self.K = K_reshaped.transpose(0, 2, 1, 3)
        self.V = V_reshaped.transpose(0, 2, 1, 3)
        
        self.S = self.Q @ self.K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        
        exp_s = np.exp(self.S - np.max(self.S, axis=-1, keepdims=True))
        self.A = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        
        self.O_heads = self.A @ self.V
        self.O_heads_transposed = self.O_heads.transpose(0, 2, 1, 3)
        self.O_heads_merged = self.O_heads_transposed.reshape(B, S, D)
        
        self.out = self.O_heads_merged @ self.W_o
        return self.out
        
    
    def backward(self, grad):
        B, S, D = grad.shape
        H = self.head_num
        d_k = self.d_k
        d_v = self.d_v
        
        O_heads_merged_2d = self.O_heads_merged.reshape(-1, D)
        grad_output_2d = grad.reshape(-1, D)
        
        self.dW_o = O_heads_merged_2d.T @ grad_output_2d
        dO_heads_merged = grad @ self.W_o.T
        
        dO_heads_transposed = dO_heads_merged.reshape(B, S, H, d_v)
        dO_heads = dO_heads_transposed.transpose(0, 2, 1, 3)
        
        dV = self.A.transpose(0, 1, 3, 2) @ dO_heads
        dA = dO_heads @ self.V.transpose(0, 1, 3, 2)
        dS = self.A * (dA - np.sum(dA * self.A, axis=-1, keepdims=True)) 
        
        dS = dS / np.sqrt(d_k)
        dQ = dS @ self.K
        dK = dS.transpose(0, 1, 3, 2) @ self.Q 
        
        dQ_proj = dQ.transpose(0, 2, 1, 3).reshape(B, S, D)
        dK_proj = dK.transpose(0, 2, 1, 3).reshape(B, S, D)
        dV_proj = dV.transpose(0, 2, 1, 3).reshape(B, S, D)
        
        X_2d = self.input_data.reshape(-1, D)
        self.dW_q = X_2d.T @ dQ_proj.reshape(-1, D)
        self.dW_k = X_2d.T @ dK_proj.reshape(-1, D)
        self.dW_v = X_2d.T @ dV_proj.reshape(-1, D)
        
        d_input = dQ_proj @ self.W_q.T + dK_proj @ self.W_k.T + dV_proj @ self.W_v.T
        return d_input
        