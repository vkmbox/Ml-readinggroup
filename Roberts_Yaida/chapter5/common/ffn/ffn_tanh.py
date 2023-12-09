import torch
from .ffn_base import FFNGmetricLogging 

class TanhNet(FFNGmetricLogging):
    def __init__(self, n0=3, nk=10, nl=3, l=3, bias_on=False):
        super().__init__(n0, nk, nl, l, bias_on)

    def forward(self, xx):
        if self.g_indices != None:
            self.GXX = dict.fromkeys(self.g_indices, None)

        #1st dimension-trainset size, 2nd dimension-layer width
        zk = torch.tensor(xx.transpose(), dtype=torch.float32)
        if self.g_indices != None:
            self.log_gmetric(zk)

        for linear in self.hidden_linears:
            zk = linear(zk)
            zk = torch.tanh(zk)
            if self.g_indices != None:
                self.log_gmetric(zk)

        zk = self.output_linear(zk)
        #if self.g_indices != None:
        #    self.record_gmetric(zk)

        return zk.detach().numpy().transpose()
