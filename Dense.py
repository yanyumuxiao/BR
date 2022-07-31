import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter


class FA_Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

     Args:
         in_features: size of each input sample
         out_features: size of each output sample
         bias: If set to ``False``, the layer will not learn an additive bias.
             Default: ``True``

     Shape:
         - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
           additional dimensions and :math:`H_{in} = \text{in\_features}`
         - Output: :math:`(N, *, H_{out})` where all but the last dimension
           are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

     Attributes:
         weight: the learnable weights of the module of shape
             :math:`(\text{out\_features}, \text{in\_features})`. The values are
             initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
             :math:`k = \frac{1}{\text{in\_features}}`
         bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                 If :attr:`bias` is ``True``, the values are initialized from
                 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                 :math:`k = \frac{1}{\text{in\_features}}`

     Examples::

         # >>> m = nn.Linear(20, 30)
         # >>> input = torch.randn(128, 20)
         # >>> output = m(input)
         # >>> print(output.size())
         torch.Size([128, 30])
     """
    __constants__ = ['in_features', 'out_features']

    def __init__(self, config, bias=True):
        super(FA_Linear, self).__init__()
        self.in_features = config.hidden
        self.out_features = config.item_count + 3

        self.n_cates = config.cate_count + 1
        self.cate_list = config.cate_list

        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features // 2))
        self.cates = Parameter(torch.Tensor(self.n_cates, self.in_features // 2))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.init_weights()
        # self.weight = Parameter(torch.cat([self.weight, self.cates[self.cate_list, :]], 1))

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, input):
        return F.linear(input, torch.cat([self.weight, self.cates[self.cate_list, :]], 1), self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
