import numpy as np

class PositionalEncoding():
    def __init__(self, max_len:int, d_model:int):
        self.max_len = max_len
        self.d_model = d_model
        pos = np.arange(0, self.max_len).reshape(-1, 1)
        even_indices = np.arange(0, self.d_model, 2)
        div_term = np.exp(-even_indices / self.d_model * np.log(10000))[np.newaxis, :]
        self.angle_rates = pos * div_term
        self.pe = np.zeros((max_len, d_model))
        self.pe[:, 0::2] = np.sin(self.angle_rates)
        self.pe[:, 1::2] = np.cos(self.angle_rates)
    
    def forward(self, x:np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        
        assert seq_len <= self.max_len, f"Input seq_len {seq_len} exceeds max_len {self.max_len}"
        assert d_model == self.d_model, f"Dimension mismatch: expected {self.d_model}, got {d_model}"
        
        return self.pe[:seq_len] + x
    
        
