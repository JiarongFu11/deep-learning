import numpy as np
from d2l import torch as d2l
from torch import nn 
import torch

true_w = torch.tensor([2, 3.4])
true_b = 4.2
batch_size = 30
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
w = torch.normal(0, 0.001, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def load_data(x, y, batch_size):
    total_len = x.shape[0]
    for i in range(batch_size, total_len, batch_size):
        batch_x = x[i - batch_size : min(i, total_len)]
        batch_y = y[i - batch_size : min(i, total_len)]
        yield batch_x, batch_y

def linear_reg(x, w, b):
    y_hat = torch.matmul(x, w) + b
    
    return y_hat

def loss_func(y_hat, y):
    return (y_hat - y) ** 2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train_model():
    num_epoches = 10
    for epoch in range(num_epoches):
        for batch_x, batch_y in load_data(features, labels, batch_size):
            net = linear_reg(batch_x, w, b)
            l = loss_func(net, batch_y)
            l.sum().backward()
            sgd([w, b], lr=0.1, batch_size=batch_size)
        
        total_loss = loss_func(linear_reg(features, w, b), labels).sum()
        print(f'epoch {epoch}: the total loss is {total_loss}')

def train_model_with_torch():
    net = nn.Sequential(nn.Linear(2,1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    loss_func = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    for epoch in range(10):
        for batch_x, batch_y in load_data(features, labels, batch_size=30):
            y_hat = net(batch_x)
            loss = loss_func(y_hat, batch_y)
            trainer.zero_grad()
            loss.backward()
            trainer.step()
        
        loss = loss_func(net(features), labels)
        print(f'epoch {epoch}: total loss {loss}')
    
    
if __name__ == '__main__':
    train_model_with_torch()