import numpy as np

class MaskedMHA():
    def __init__(self, input_dim, d_model, d_k, d_v, head_num):
        self.input_dim = input_dim
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.head_num = head_num
        self._init_params()
        
    def _init_params(self,):
        assert self.d_model == self.d_k * self.head_num
        limit = np.sqrt(6 / (self.d_model + self.d_model))
        
        self.Wq = np.random.uniform(-limit, limit, size=(self.d_model, self.d_model))
        self.Wk = np.random.uniform(-limit, limit, size=(self.d_model, self.d_model))
        self.Wv = np.random.uniform(-limit, limit, size=(self.d_model, self.d_model))
        self.Wo = np.random.uniform(-limit, limit, size=(self.d_model, self.d_model))
    
    def forward(self, input_data:np.ndarray) -> np.ndarray:
        self.input_data = input_data
        B,S,D = input_data.shape
        H = self.head_num
        
        self.Masked = np.where(np.triu(np.ones((S, S))) == 1, -np.inf, 0)
        self.Masked = self.Masked[np.newaxis, np.newaxis, :, :]
        
        self.Q_proj = self.input_data @ self.Wq
        self.K_proj = self.input_data @ self.Wk
        self.V_proj = self.input_data @ self.Wv
        
        self.Q_reshape = self.Q_proj.reshape(B, S, H, self.d_k)
        self.K_reshape = self.K_proj.reshape(B, S, H, self.d_k)
        self.V_reshape = self.V_proj.reshape(B, S, H, self.d_k)
        
        self.Q = self.Q_reshape.transpose(0, 2, 1, 3)
        self.K = self.K_reshape.transpose(0, 2, 1, 3)
        self.V = self.V_reshape.transpose(0, 2, 1, 3)
        
        self.S = self.Q @ self.K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k) + self.Masked
        exp_S = np.exp(self.S - np.max(self.S, axis=-1, keepdims=True))
        self.A = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
        
        self.Oheads = self.A @ self.V
        self.Oheads_transpose = self.Oheads.transpose(0, 2, 1, 3)
        self.Oheads_merged = self.Oheads_transpose.reshape(B, S, D)
        
        out = self.Oheads_merged @ self.Wo

        return out
    
    def backward(self, grad:np.ndarray) -> np.ndarray:
        pass
        