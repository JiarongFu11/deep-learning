import numpy as np

class LinearNormalization():
    def __init__(self, d_model:int, eps:float=1e-5):
        self.d_model = d_model
        self.eps = eps
        self._init_params()
    
    def _init_params(self) -> None:
        self.gamma = np.ones(self.d_model)
        self.beta = np.zeros(self.d_model)
    
    def forward(self, input_data:np.ndarray) -> np.ndarray:
        self.cache = input_data
        self.mean_ = np.mean(input_data, axis=-1, keepdims=True)
        self.var_ = np.var(input_data, axis=-1, keepdims=True)
        self.norm_data = (input_data - self.mean_) / (np.sqrt(self.var_ + self.eps))
        return self.gamma * self.norm_data + self.beta
    
    def backward(self, grad:np.ndarray) -> np.ndarray:
        self.gamma_grad = np.sum(self.norm_data * grad, axis=(0, 1))
        self.beta_grad = np.sum(grad, axis=(0, 1))
        
        D = self.d_model
        term1 = D * grad * self.gamma
        term2 = np.sum(grad * self.gamma, axis=-1, keepdims=True)
        term3 = self.norm_data * np.sum(grad * self.gamma * self.norm_data, axis=-1, keepdims=True)
        
        grad_input = (term1 - term2 - term3) / (D * np.sqrt(self.var_ + self.eps))
        
        return grad_input
    