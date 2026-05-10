import numpy as np

class Attention():
    def __init__(self, input_dim, d_model, d_k, d_v):
        self.input_dim = input_dim
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self._init_params()
    
    def _init_params(self,):
        self.W_q = np.random.uniform(
            -np.sqrt(6 / (self.d_model + self.d_k)), np.sqrt(6 / (self.d_model + self.d_k)), 
            size=(self.input_dim, self.d_k)
            )
        
        self.W_k = np.random.uniform(
            -np.sqrt(6 / (self.d_model + self.d_k)), np.sqrt(6 / (self.d_model + self.d_k)),
            size=(self.input_dim, self.d_k)
            )
        
        self.W_v = np.random.uniform(
            -np.sqrt(6 / (self.d_model + self.d_v)), np.sqrt(6 / (self.d_model + self.d_v)),
            size=(self.input_dim, self.d_v)
            )
        
    def forward(self, input_data:np.ndarray):
        self.Q = input_data @ self.W_q
        self.K = input_data @ self.W_k
        self.V = input_data @ self.W_v
        self.S = (self.Q @ self.K.swapaxes(-1, -2)) / np.sqrt(self.d_k)
        exp_S = np.exp(self.S - np.max(self.S, axis=-1, keepdims=True))
        self.A = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
        self.O = self.A @ self.V
        
        return self.O
        
        
    
    def backward(self, grad):
        self.grad_V = self.A.T @ grad
        self.grad_A= grad @ self.V.T
        self.grad_S = self.A * (self.grad_A - np.sum(self.A * self.grad_A, axis=-1, keepdims=True))
        self.grad_Q = (self.grad_S @ self.K) / np.sqrt(self.d_k)
        self.grad_k = self.grad_S.swapaxes(-1,-2) @ self.Q / np.sqrt(self.d_k)
        
        