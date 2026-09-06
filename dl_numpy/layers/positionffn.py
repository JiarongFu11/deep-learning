import numpy as np

class PositionwiseFeedForward():
    def __init__(self, d_model:int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        self._init_params()
    
    def _init_params(self,):
        self.W1 = np.random.normal(scale=np.sqrt(2.0 / self.d_model),size=(self.d_model, self.d_ff))
        self.W2 = np.random.normal(loc=0, scale=np.sqrt(2.0 / self.diff), size=(self.d_ff, self.d_model))
        
        self.b1 = np.zeros(self.d_ff)       
        self.b2 = np.zeros(self.d_model)
        
    def forward(self, x:np.ndarray) -> np.ndarray:
        self.x = x
        
        self.h1 = np.maximum(self.x @ self.W1 + self.b1, 0)
        self.h2 = self.h1 @ self.W2 + self.b2
        
        return self.h2
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        B, S, D = grad.shape
        
        grad_flat = grad.reshape(-1, self.d_model)
        h1_flat = self.h1.reshape(-1, self.d_ff)
        x_flat = self.x.reshape(-1, self.d_model)
        
        self.dW2 = h1_flat @ grad_flat
        self.db2 = np.sum(grad, axis=(0,1))
        dh1 = grad @ self.W2.T
        
        dz1 = dh1 * (self.h1 > 0)
        dz1_flat = dz1.reshape(-1, self.d_ff)            
        
        self.dW1 = x_flat.T @ dz1_flat                   
        self.db1 = np.sum(dz1, axis=(0, 1))     
        dx = dz1 @ self.W1.T    
        
        return dx
        