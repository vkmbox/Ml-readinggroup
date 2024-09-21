import numpy as np
from torch import nn

from common.ffn.ffn_relu import ParametricReLUNet
from common.ffn_model.minst.ffn_minst_util import StepCalculatorBase

import logging

class MNISTReLU(ParametricReLUNet):
    def __init__(self, meta):
        super().__init__(n0=meta.input_dim,nk=0,nl=meta.output_dim,l=0, bias_on=True)
        self.meta = meta
        self.input_fc = nn.Linear(meta.input_dim, meta.input_width)
        self.hidden_fc = nn.Linear(meta.input_width, meta.hidden_width)
        self.output_fc = nn.Linear(meta.hidden_width, meta.output_dim)

    def forward_(self, x):
        return self.forward(x)

    def forward(self, x):
        #x = [batch size, height * width]
        h_1 = self.PReLU(self.input_fc(x))
        #h_1 = [batch size, INPUT_WIDTH]
        h_2 = self.PReLU(self.hidden_fc(h_1))
        #h_2 = [batch size, HIDDEN_WIDTH]
        y_pred = self.output_fc(h_2)
        #y_pred = [batch size, output dim]
        return y_pred
    
    def init_weights(self, cb=0.0, cw=1.0):
        if self.get_log_level() == "debug":
            logging.debug("FeedForwardNet weights initialisation with cb={}, cw={}".format(cb, cw))

        #Weight initialisation as in 2.19, 2.20
        self.cb, self.cw = cb, cw
        self.init_linear_weights(self.input_fc, self.bias_on, cb, cw/self.input_fc.in_features)
        self.init_linear_weights(self.hidden_fc, self.bias_on, cb, cw/self.hidden_fc.in_features)
        self.init_linear_weights(self.output_fc, self.bias_on, cb, cw/self.output_fc.in_features)

    def grad_zero(self):
        self.input_fc.weight.grad.zero_()
        self.input_fc.bias.grad.zero_()
        self.hidden_fc.weight.grad.zero_()
        self.hidden_fc.bias.grad.zero_()
        self.output_fc.weight.grad.zero_()
        self.output_fc.bias.grad.zero_()

    def save_txt(self, dir_name):
        np.savetxt(dir_name + '/input_weight.out', self.input_fc.weight.detach().numpy(), delimiter=',')
        np.savetxt(dir_name + '/input_bias.out', self.input_fc.bias.detach().numpy(), delimiter=',')
        np.savetxt(dir_name + '/hidden_weight.out', self.hidden_fc.weight.detach().numpy(), delimiter=',')
        np.savetxt(dir_name + '/hidden_bias.out', self.hidden_fc.bias.detach().numpy(), delimiter=',')
        np.savetxt(dir_name + '/output_weight.out', self.output_fc.weight.detach().numpy(), delimiter=',')
        np.savetxt(dir_name + '/output_bias.out', self.output_fc.bias.detach().numpy(), delimiter=',')

    def init_weights_txt(self, dir_name):
        self.init_linear_zeros(self.input_fc, self.bias_on)
        self.init_linear_zeros(self.hidden_fc, self.bias_on)
        self.init_linear_zeros(self.output_fc, self.bias_on)
        calc0 = StepCalculatorBase()
        calc0.delta_weight_00 = np.loadtxt(dir_name + '/input_weight.out', delimiter=',')
        calc0.delta_bias_00 = np.loadtxt(dir_name + '/input_bias.out', delimiter=',')
        calc0.delta_weight_01 = np.loadtxt(dir_name + '/hidden_weight.out', delimiter=',')
        calc0.delta_bias_01 = np.loadtxt(dir_name + '/hidden_bias.out', delimiter=',')
        calc0.delta_weight_02 = np.loadtxt(dir_name + '/output_weight.out', delimiter=',')
        calc0.delta_bias_02 = np.loadtxt(dir_name + '/output_bias.out', delimiter=',')        
        calc0.do_step0(self)
