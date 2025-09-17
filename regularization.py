import torch
import torch.nn as nn
def l1_regularization(model, lambda_1):
    l1_loss = 0
    for param in model.parameter():
        l1_loss += torch.norm(param, 1)
    return l1_loss * lambda_1

def model():
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

def l2_regularization(model, lambda_2):
    #the weight decay parameter is used to add the l2 norm into the loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
    #or only using norm to weight instead of bias
    optimizer = torch.optim.SGD(
        [{'params': model.weight, 'weight_decay': lambda_2},
         {'params': model.bias}],
        lr = 0.01
    )
    #besides, we can set the lr for different parameters
    optimizer = torch.optim.SGD(
        [{'params': model.weight, 'lr': 0.01, 'weight_decay': lambda_2},
         {'params': model.bias, 'lr': 0.01, 'weight_decay': lambda_2}]
    )
    return optimizer

