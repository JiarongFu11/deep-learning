import torch

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(y_hat.shape[0]), y])

y = torch.tensor(range(10))
y_hat = torch.normal(0, 1, size=(y.shape[0], max(y) + 1))

def sofmax(x):
    return torch.exp(x) / torch.exp(x).sum(1, keepdim=True)

prob_label = sofmax(y_hat)
print(prob_label)
loss = cross_entropy(prob_label, y)

def accuracy(y_hat, y):
    y_hat = y_hat.argmax(1)
    cmp = y_hat.type(y.dtype) == y
    print(cmp.type(y.dtype).sum())
    return cmp.type(y.dtype).sum() / y_hat.shape[0]

print(accuracy(y_hat, y))