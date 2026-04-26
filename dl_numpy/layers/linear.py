import torch
import torch.nn as nn
import numpy as np

class LinearLayer():
    def __init__(self, input_dim:int, output_dim:int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._init_params()
        
    def _init_params(self,):
        scale = 1 / np.sqrt(self.input_dim)
        self.W = np.random.normal(size=(self.input_dim, self.output_dim), loc=0, scale=scale)
        self.b = np.random.normal(size=(self.output_dim), loc=0, scale=1)
    
    def forward(self, input_data:np.ndarray) -> np.ndarray:
        """calculate Y = XW + b """
        self.cache = input_data
        return input_data @ self.W + self.b
    
    def backward(self, grad:np.ndarray) -> np.ndarray:
        """calculate gradient"""
        self.W_grad = self.cache.T @ grad
        self.b_grad = np.sum(grad, axis=0)
        grad_input = grad @ self.W.T
        
        return grad_input

class CustomLinear(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(output_dim, input_dim))
        self.b = nn.Parameter(torch.empty(output_dim,))
        
        self._init_params()
    
    def _init_params(self,):
        nn.init.kaiming_normal_(self.W)
        nn.init.zeros_(self.b)
    
    def forward(self, input_data):
        return input_data @ self.W.T + self.b
        
    

input_data = np.random.normal(size=(100, 10))
output_data = LinearLayer(10, 5).forward(input_data)
print(output_data)