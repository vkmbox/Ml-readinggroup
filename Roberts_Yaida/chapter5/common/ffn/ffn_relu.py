import torch
#import torch.nn.functional as F
from torch import Tensor
from .ffn_base import FFNGmetricLogging 

class ParametricReLUNet(FFNGmetricLogging):
    def __init__(self, n0=3, nk=10, nl=3, l=3, bias_on=False):
        super().__init__(n0, nk, nl, l, bias_on)
        self.slope_positive = None
        self.slope_negative = None

    def set_slopes(self, slope_positive = 1.0, slope_negative = 0.25):
        self.slope_positive = slope_positive
        self.slope_negative = slope_negative

    def PReLU(self, input: Tensor) -> Tensor:
        input = torch.where(input >= 0, self.slope_positive * input, self.slope_negative * input)
        return input

    def PReLUz(self, input: float) -> float:
        return self.slope_positive * input if input >= 0 else self.slope_negative * input

    '''
    def PReLUa(self, input):
        print(input.shape)
        return [self.slope_positive * item if item >= 0 else self.slope_negative * item for item in input]
    '''
    def forward(self, xx):
        #1st dimension-trainset size, 2nd dimension-layer width
        z0 = torch.tensor(xx.transpose(), dtype=torch.float32)
        z_out = self.forward_(z0)
        return z_out.detach().numpy().transpose()
    
    def forward_(self, zk):
        if self.slope_positive == None:
            raise Exception("To use forward set slopes with call ParametricReLUNet.set_slopes(...)")

        if self.g_indices != None:
            self.GXX = dict.fromkeys(self.g_indices, None)
        if self.pre_indices != None:
            self.PRE = dict.fromkeys(self.pre_indices, None)

        self.trigger_on_forward_step_activ_callbacks(zk.detach().numpy())

        for linear in self.hidden_linears:
            zk = linear(zk)
            self.trigger_on_forward_step_preactiv_callbacks(zk.detach().numpy())
            zk = self.PReLU(zk)
            #zk = F.relu(zk)
            self.trigger_on_forward_step_activ_callbacks(zk.detach().numpy())

        zk = self.output_linear(zk)
        self.trigger_on_forward_step_preactiv_callbacks(zk.detach().numpy())

        return zk #.detach().numpy().transpose()
