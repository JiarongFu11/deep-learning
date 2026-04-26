import numpy as np

np.random.seed(40)

input_data = np.random.randn(1, 3, 277, 277)

class ConvolutionLayer():
    def __init__(self, width:int, c_in:int, c_out:int, kernal_width:int, stride:int=1, padding:int=0, pad_width:int=2):
        self.c_out = c_out
        self.c_in = c_in
        self.stride = stride
        self.kernal_w = kernal_width
        self.padding = padding
        self.pad_width = pad_width
        
        self.W = np.random.normal(0, 0.01, (self.c_out, self.c_in, self.kernal_w, self.kernal_w))
        self.b = np.ones((self.c_out))
        
    
    def forward(self, input_data) -> np.ndarray:
        
        input_data = np.pad(input_data, pad_width=((0,0), (0, 0), (self.pad_width, self.pad_width), (self.pad_width, self.pad_width)), mode='constant', constant_values=0)
        self.N, self.c_in, self.width, self.width = input_data.shape
        output_w = int((self.width - self.kernal_w) / self.stride) + 1
        self.output = np.zeros((self.N, self.c_out, output_w, output_w))

        for n in range(self.N):
            for c_o in range(self.c_out):
                o_h = 0
                for i in range(0, self.width - self.kernal_w + 1, self.stride):
                    o_w = 0
                    for j in range(0, self.width - self.kernal_w + 1, self.stride):
                        for c_i in range(self.c_in):
                            self.output[n, c_o, o_h, o_w] += np.sum(input_data[n, c_i, i : i + self.kernal_w, j : j + self.kernal_w] * self.W[c_o, c_i, :, :])
                        self.output[n, c_o, o_h, o_w] += self.b[c_o]
                        o_w += 1
                    o_h += 1
        
        return self.output


                    
                    
output = ConvolutionLayer(277, 3, 2, 3).forward(input_data)
print(output)     

    
