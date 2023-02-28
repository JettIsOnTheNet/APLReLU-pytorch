# APLReLU
Adaptive Piecewise Linear activation function in pytorch

```

import torch
import torch.nn.functional as F

class APLReLU(torch.nn.Module):
    def __init__(self, num_parameters=2, init=0.25):
        super(APLReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = torch.nn.Parameter(torch.Tensor(num_parameters).fill_(init))
        self.bias = torch.nn.Parameter(torch.Tensor(num_parameters).zero_())
    
    def forward(self, x):
        out = F.relu(x - self.bias[0]) * self.weight[0]
        for i in range(1, self.num_parameters):
            mask = x >= self.bias[i]
            out += mask * (x - self.bias[i]) * self.weight[i]
        return out

```
