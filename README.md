# deep-learning

My notes-in-code for learning deep learning. The idea is to write the basic pieces by hand first (NumPy, or raw PyTorch tensors with autograd), then compare against the built-in PyTorch versions.

Still a work in progress. Some files are placeholders that are empty for now.

## Layout

```
deep-learning/
├── dl_numpy/                 # hand-written implementations
│   ├── layers/
│   │   ├── linear.py         #   linear layer with forward/backward in NumPy, plus a custom nn.Module version
│   │   ├── cnn.py            #   2D convolution forward pass with plain loops
│   │   ├── dropout.py        #   dropout from scratch and with nn.Dropout
│   │   ├── sofmax.py         #   softmax, cross entropy, accuracy
│   │   ├── activation.py     #   (todo)
│   │   ├── attention.py      #   (todo)
│   │   └── normalization.py  #   (todo)
│   ├── loss/
│   │   ├── regularization.py #   L1 penalty, L2 via weight_decay
│   │   └── cross_entropy.py  #   (todo)
│   ├── models/
│   │   ├── MLP.py            #   MLP trained from scratch: init, forward, softmax loss, SGD
│   │   ├── resnet.py         #   (todo)
│   │   └── transformer.py    #   (todo)
│   └── optim/
│       ├── sgd.py            #   (todo)
│       └── adam.py           #   (todo)
└── dl_torch/                 # same things using PyTorch modules
    ├── layers/cnn_torch.py   #   nn.Conv2d version of the conv layer
    ├── data/ models/ utils/  #   (todo)
    └── train.py              #   (todo)
```

## Setup

```bash
pip install numpy torch d2l
```

`d2l` is only used in `dropout.py`.

## Running

Most files are standalone scripts that run a small example on random data:

```bash
python dl_numpy/models/MLP.py      # trains a 2-layer MLP on random data, prints loss per epoch
python dl_numpy/layers/linear.py
python dl_numpy/layers/cnn.py      # slow on purpose, it's a naive loop
python dl_torch/layers/cnn_torch.py
```

## Plan

- Fill in activations, normalization, attention
- SGD and Adam optimizers
- ResNet and Transformer
- Backward pass for the conv layer
- A proper training script in `dl_torch/`