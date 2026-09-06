import numpy as np

class Conv2d():
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernal_size, 
            padding_size:int = 0, 
            stride:int = 1,
        ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernal_size = kernal_size
        self.padding_size = padding_size
        self.stride = stride
        
        self._init_weights()
    
    def _init_weights(self, ):
        fan_in = self.C_in * self.kernel_size * self.kernel_size
        self.Weight = np.random.normal(size=(self.out_channels, self.in_channels, self.K, self.K), loc=0, scale=np.sqrt(2 /fan_in))
        self.b = np.zeros(size=(self.out_channels))
        
    def forward(self, input_data:np.ndarray):
        self.input_data = input_data
        N, c_in, H_in, W_in = input_data.shape
        
        assert c_in == self.in_channels, f"Expected {self.in_channels} channels, got {c_in}"
        
        H_out = (H_in + 2 * self.padding_size - self.kernal_size) / self.stride + 1
        W_out = (W_in + 2 * self.padding_size - self.kernal_size) / self.stride + 1
        
        self.X_pad = np.pad(
            input_data,
            pad_width=((0,0), (0,0), (self.padding_size, self.padding_size), (self.padding_size, self.padding_size)),
            mode='constant'
        )
        
        Y = np.zeros((N, self.out_channels, H_out, W_out))
        
        for H in range(self.H_out):
            for W in range(self.W_out):
                x_slice = self.X_pad[:, :, H * self.stride:H * self.stride + self.kernal_size, W * self.stride:W * self.kernal_size + self.kernal_size]
                for c_out in range(self.out_channels):
                    Y[:, c_out, H, W] = np.sum(x_slice * self.Weight[c_out], axis=(1,2,3)) + self.b[c_out]
        
        return Y
        
    def backward(self, grad_out:np.ndarray):
        N, C_out, H_out, W_out = grad_out.shape
    
        db = np.sum(grad_out, axis=(0, 2, 3))
        dW = np.zeros_like(self.Weight)
        dX_pad = np.zeros_like(self.X_pad)
        
        for h in range(H_out):
            h_start = h * self.stride
            h_end = h_start + self.kernel_size
            
            for w in range(W_out):
                w_start = w * self.stride
                w_end = w_start + self.kernel_size
                
                x_slice = self.X_pad[:, :, h_start:h_end, w_start:w_end]
                
                for c_out in range(self.out_channels):
                    dY_val = grad_out[:, c_out, h, w, np.newaxis, np.newaxis, np.newaxis]
                    dW[c_out] += np.sum(x_slice * dY_val, axis=0)
                    
                    dX_pad[:, :, h_start:h_end, w_start:w_end] += dY_val[:, :, 0, 0] * self.Weight[c_out]
    
        if self.padding_size > 0:
            dX = dX_pad[:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
        else:
            dX = dX_pad
        
        return dX
        