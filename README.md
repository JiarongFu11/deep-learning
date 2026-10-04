# deep-learning

My notes-in-code for learning deep learning. Most of the pieces are written by hand in NumPy, with forward and backward passes, and then rebuilt with PyTorch modules to compare.

Still a work in progress.

## Layout

```
deep-learning/
├── dl_numpy/                        # hand-written, NumPy
│   ├── layers/
│   │   ├── linear.py                #   linear layer, forward + backward
│   │   ├── activation.py            #   Sigmoid, Tanh, ReLU
│   │   ├── cnn.py                   #   first conv attempt, naive loops, forward only
│   │   ├── Conv2d.py                #   conv layer with forward + backward
│   │   ├── batchnormalization.py    #   BatchNorm for 2D and 4D inputs, running stats
│   │   ├── linear_normalization.py  #   LayerNorm
│   │   ├── attention.py             #   single-head scaled dot-product attention
│   │   ├── multi_attention.py       #   multi-head attention, forward + backward
│   │   ├── masked_mha.py            #   causal masked MHA (forward only so far)
│   │   ├── positional_encoding.py   #   sinusoidal positional encoding
│   │   └── positionffn.py           #   position-wise feed-forward
│   ├── loss/
│   │   └── regularization.py        #   L1 penalty, L2 via weight_decay (PyTorch)
│   └── models/
│       ├── MLP.py                   #   MLP trained from scratch with autograd tensors + manual SGD
│       └── transformer_encoder_block.py  # pre-LN encoder block built from the layers above
└── dl_torch/                        # same ideas with PyTorch modules
    ├── layers/
    │   ├── multiheadattention.py    #   MHA with optional mask
    │   ├── positional_encoding.py
    │   └── resnet.py                #   ResNet BasicBlock
    ├── models/
    │   └── transformer_encoder.py   #   pre-LN encoder block using nn.MultiheadAttention
    ├── data/ utils/                 #   empty for now
    └── train.py                     #   empty for now
```

## Setup

```bash
pip install numpy torch
```

## Running

A few files run a small example on random data:

```bash
python dl_numpy/models/MLP.py      # 2-layer MLP, prints loss per epoch
python dl_numpy/layers/linear.py
python dl_numpy/layers/cnn.py      # slow, it's a naive loop
```

`transformer_encoder_block.py` uses relative imports, so import it as a package from the repo root:

```python
from dl_numpy.models.transformer_encoder_block import TransformerEncoderBlock
```

## Plan

- Backward pass for masked MHA, and fix the backward in `attention.py`
- Optimizers (SGD, Adam) as their own module
- A full Transformer and ResNet
- Data loading and a training script in `dl_torch/`
