import numpy as np
from abc import ABC 

class Activation(ABC):   
    def __init__(self,):
        pass
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Sigmoid(Activation):
    def __init__(self,):
        pass
    
    def forward(self, input_data):
        self.cache = np.clip(input_data, -500, 500)
        self.output_data = 1 / (1 + np.exp(-self.cache))
        return self.output_data
    
    def backward(self, grad_output):
        return grad_output * self.output_data * (1 - self.output_data)
    
class Tanh(Activation):
    def __init__(self, ):
        pass
    
    def forward(self, input_data):
        self.cache = np.clip(input_data, -500, 500)
        self.output_data = (np.exp(self.cache) - np.exp(-self.cache)) / (np.exp(self.cache) + np.exp(-self.cache))
        return self.output_data
    
    def backward(self, grad_output):
        return grad_output * (1 - self.output_data ** 2)
    
class Rulu(Activation):
    def __init__(self,):
        pass
    
    def forward(self, input_data):
        self.cache = input_data
        return np.where(input_data > 0, input_data, 0)
    
    def backward(self, grad_output):
        return grad_output * np.where(self.cache > 0, 1, 0)