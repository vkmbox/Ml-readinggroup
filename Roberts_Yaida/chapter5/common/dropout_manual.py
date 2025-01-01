import torch
import math


class DropoutManual(torch.autograd.Function):

    maps = {}

    """
    input - input tensor
    prob - required probability
    toss - "-1" to skip or index to generate map
    train - "-1" to skip or index to use map
    """
    @staticmethod
    def forward(ctx, input, prob, toss, train):
        map = None
        if toss >= 0:
            map = torch.bernoulli(torch.full_like(input, prob))/(1-prob)
            DropoutManual.maps[toss] = map
        elif train >= 0:
            map = DropoutManual.maps[train]
        else:
            map = torch.ones_like(input)

        ctx.save_for_backward(input, map)
        return input*map

    @staticmethod
    def backward(ctx, grad_output):
        _, map = ctx.saved_tensors
        return grad_output*map, None, None, None
