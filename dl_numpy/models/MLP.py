import torch

def init_para(param_size_l):
    param_list = []
    for input_size, output_size in param_size_l:
        w = torch.normal(0, 0.01, size=(input_size, output_size), requires_grad=True)
        b = torch.zeros(output_size, requires_grad=True)
        param_list.append([w, b])
    
    return param_list

def data_iter(features, labels, batch_size):
    data_len = features.shape[0]
    for i in range(0, data_len, batch_size):
        s_i = i
        e_i = min(i + batch_size, data_len)
        yield features[s_i:e_i], labels[s_i:e_i]

def linear_reg(x, w, b):
    return torch.matmul(x, w) + b

def relu_func(output):
    zero_matrix = torch.zeros_like(output)
    return torch.max(zero_matrix, output)

def model(params, input):
    for params_i, (w, b) in enumerate(params):
        output = linear_reg(input, w, b)
        if params_i < len(params) - 1:
            output = relu_func(output)
        input = output
    
    return output

def softmax(y_label):
    max_axis = torch.max(y_label, dim=1,keepdim=True)[0]
    exp_y = torch.exp(y_label - max_axis)
    return exp_y / exp_y.sum(dim=1, keepdim=True)
    
def loss_func(y_label, y):
    prob_matrix = softmax(y_label)
    return - torch.log(prob_matrix[range(y_label.shape[0]), y])

def optim(params, lr, batch_size):
    with torch.no_grad():
        for w, b in params:
            w -= lr * w.grad / batch_size
            b -= lr * b.grad / batch_size
            
            w.grad.zero_()
            b.grad.zero_()

def train(features, labels):
    epochs = 10
    batch_size = 30
    lr = 0.01
    num_class = 3
    params = init_para([[features.shape[1], 10], [10, num_class]])
    
    for epoch in range(epochs):
        for batch_x, batch_y in data_iter(features, labels, batch_size):
            output = model(params, batch_x)
            
            loss = loss_func(output, batch_y)
            loss.sum().backward()
            optim(params, lr, batch_size)
        
        with torch.no_grad():
            total_loss = loss_func(model(params, features), labels).sum()
            print(f'total loss: {total_loss}')
            
if __name__ == "__main__":
    n_samples = 1000
    n_features = 20
    n_classes = 3
    
    features = torch.randn(n_samples, n_features)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    train(features, labels)