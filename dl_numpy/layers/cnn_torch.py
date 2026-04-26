import torch 
import torch.nn as nn

conv_tensor = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1,padding=2)
nn.init.normal_(conv_tensor.weight, mean=0, std=0.01)
nn.init.constant_(conv_tensor.bias, 1)

input_tensor = torch.randn(1,3,277,277)
output_tensor = conv_tensor(input_tensor)
print(output_tensor)