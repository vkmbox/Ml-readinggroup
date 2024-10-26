import numpy as np
import torch
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
    
    #--NTK-values calculated as in (8.12), based on experiments by Zhang Allan--
    def forward_ntk(self, x):
        '''forward when also calculation ntk;
        x ~ n_samples * input_dim;
        lw: lambda_w;
        lb: lambda_b;
        returns y and ntk of the last layer'''
        meta = self.meta
        #1st layer (the "input layer")
        #x = self.flatten(x)
        logging.info("##Calculating NTK input layer")
        with torch.no_grad():
            H = meta.lw / meta.input_dim * torch.matmul(x.to('cpu'), x.to('cpu').T) + meta.lb
        y = self.PReLU(self.input_fc(x))
        with torch.no_grad():
            yc = y.to('cpu')
            yp = self.activation_derivative(yc)
        #2nd layer (the hidden layer)
            logging.info("##Calculating NTK 2nd layer")
            w = self.hidden_fc.weight.to('cpu')
            H = H * yp.T[:, None, :] * yp.T[:, :, None]
            H = w[:, :, None, None] * H
            H = torch.tensordot(w, H, ([1], [1]))
            Hd = torch.movedim(torch.diagonal(H), -1, 0) 
            Hd += meta.lw / meta.input_width * torch.matmul(yc, yc.T) + meta.lb
        y = self.PReLU(self.hidden_fc(y))
        with torch.no_grad():
            yc = y.to('cpu')
            yp = self.activation_derivative(yc)
        #3rd layer (the output layer)
            logging.info("##Calculating NTK 3rd layer")
            w = self.output_fc.weight.to('cpu')
            H *= yp.T[:, None, :, None]
            H *= yp.T[:, None, :]
            H = torch.tensordot(w, H, ([1], [1]))
            H = torch.tensordot(w, H, ([1], [1])) #???
            Hd = torch.movedim(torch.diagonal(H), -1, 0) 
            Hd += meta.lw / meta.hidden_width * torch.matmul(yc, yc.T) + meta.lb
        y = self.output_fc(y)
        logging.info("##Calculating NTK finished")
        return y, H
    
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

    def activation_derivative(self, xx):
        '''calculate the derivative of relu'''
        return torch.where(xx>0, self.slope_positive, self.slope_negative)

