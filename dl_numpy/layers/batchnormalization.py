import numpy as np

class BatchNormalization():
    def __init__(self, num_features:int, momentum:float=0.1):
        self.init_params(num_features, momentum)
    
    def init_params(self, num_features, momentum):
        self.gamma = np.ones((1, num_features, 1, 1))
        self.b = np.zeros((1, num_features, 1, 1))
        
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_std = np.ones((1, num_features, 1, 1))
        self.momentum = momentum
        self.eps = 1e-5
    
    def forward(self, input_data:np.ndarray, train=True) -> np.ndarray:
        
        if input_data.ndim == 4:
            self.axis = (0, 2, 3)
        else:
            self.axis = 0
            if self.gamma.ndim == 4:
                self.gamma = self.gamma.reshape(-1)
                self.beta = self.beta.reshape(-1)
                self.running_mean = self.running_mean.reshape(-1)
                self.running_std = self.running_std.reshape(-1)
        
        if train:
            mean_ = np.mean(input_data, axis=self.axis, keepdims=True)
            std_ = np.std(input_data, axis=self.axis, keepdims=True)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * std_
            
            self.norm_data = (input_data - mean_) / (std_ + 1e-5)
            self.cache = input_data
        else:
            self.norm_data = (input_data - self.running_mean) / (self.running_std + 1e-5)
        
        return self.norm_data * self.gamma + self.b
    
    def backward(self, grad:np.ndarray) -> np.ndarray:
        self.gamma_grad = np.sum(grad * self.norm_data, axis=self.axis)
        self.b_grad = np.sum(grad, axis=self.axis)
        N = np.prod([grad.shape[i] for i in (self.axis if isinstance(self.axis, tuple) else [self.axis])])
        
        grad_input = (self.gamma / (N * np.std(self.cache, axis = 0) + 1e-5)) * \
            (N * grad - np.sum(grad, axis=0, keepdims=True) - self.norm_data * np.sum(grad * self.norm_data, axis=0, keepdims=True))
        
        return grad_input

