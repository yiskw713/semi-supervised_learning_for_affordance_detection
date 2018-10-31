'''
Copyright (c) 2017 Kazuto Nakashima
Released under the MIT license
https://github.com/kazuto1011/deeplab-pytorch/blob/master/LICENSE
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class MSC(nn.Module):
    """Multi-scale inputs"""

    def __init__(self, scale, pyramids=[0.5, 0.75]):
        super(MSC, self).__init__()
        self.scale = scale
        self.pyramids = pyramids

    def forward(self, x):
        # Original
        logits = self.scale(x)
        interp = lambda l: F.interpolate(l, size=logits.shape[2:], mode="bilinear")

        # Scaled
        logits_pyramid = []
        for p in self.pyramids:
            size = [int(s * p) for s in x.shape[2:]]
            h = F.interpolate(x, size=size, mode="bilinear")
            logits_pyramid.append(self.scale(h))

        # Pixel-wise max
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        if self.training:
            return logits # [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max