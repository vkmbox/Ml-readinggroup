import torch
from torch import Tensor
from .ffn_base import FFNGmetricLogging 

'''
def poly(xx):
    return xx + xx**2/2 - xx**3/8 + xx**5/4
'''
class PolynomialActivation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input + input ** 2/2 + input ** 3/8 + input ** 5/4

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (1 + input + 3*input ** 2/8 + 5*input ** 4/4)

class AnyActivationNet(FFNGmetricLogging):
    def __init__(self, act, n0=3, nk=10, nl=3, l=3, bias_on=False):
        super().__init__(n0, nk, nl, l, bias_on)
        self.act = act

#    def activation(self, input: Tensor) -> Tensor:
#        input = input.apply_(self.act) #torch.where(input >= 0, self.slope_positive * input, self.slope_negative * input)
#        return input

    def forward(self, xx):
        if self.g_indices != None:
            self.GXX = dict.fromkeys(self.g_indices, None)
        if self.pre_indices != None:
            self.PRE = dict.fromkeys(self.pre_indices, None)

        #1st dimension-trainset size, 2nd dimension-layer width
        zk = torch.tensor(xx.transpose(), dtype=torch.float32)
        self.trigger_on_forward_step_activ_callbacks(zk.detach().numpy())

        for linear in self.hidden_linears:
            zk = linear(zk)
            self.trigger_on_forward_step_preactiv_callbacks(zk.detach().numpy())
            zk = PolynomialActivation.apply(zk)
            self.trigger_on_forward_step_activ_callbacks(zk.detach().numpy())

        zk = self.output_linear(zk)
        self.trigger_on_forward_step_preactiv_callbacks(zk.detach().numpy())

        return zk.detach().numpy().transpose()
