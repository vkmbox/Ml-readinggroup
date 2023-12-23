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
        self.trigger_on_forward_step_activ_callbacks(zk.detach().numpy())

        for linear in self.hidden_linears:
            zk = linear(zk)
            self.trigger_on_forward_step_preactiv_callbacks(zk.detach().numpy())
            zk = torch.tanh(zk)
            self.trigger_on_forward_step_activ_callbacks(zk.detach().numpy())

        zk = self.output_linear(zk)
        self.trigger_on_forward_step_preactiv_callbacks(zk.detach().numpy())

        return zk.detach().numpy().transpose()
