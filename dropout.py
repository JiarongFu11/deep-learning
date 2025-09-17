import torch
import torch.nn as nn
from d2l import torch as d2l
def dropout(x, pro):
    random_matrix = torch.rand(x.shape)
    random_matrix = (random_matrix > pro).float
    #the reason why divide (1- p) is we should maintain the expectation of output
    return random_matrix * x / (1 - pro)

class Net(nn.Module):
    def __init__(self, input_shape, hidden_1, hidden_2, output_shape, is_training=True):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(input_shape, hidden_1)
        self.lin2 = nn.Linear(hidden_1, hidden_2)
        self.lin3 = nn.Linear(hidden_2, output_shape)
        self.relu = nn.ReLU()
        
        self.training = is_training
    
    def forward(self, x):
        H1 = self.relu(self.lin1(x))
        if self.training:
            H1 = dropout(H1)
            
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout(H2)
        
        output = self.lin3(H2)
        return output

#the simple way to realize the dropout
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256,10)
)

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weight)
trainer = torch.optim.SGD(net.parameters(), lr=0.01)