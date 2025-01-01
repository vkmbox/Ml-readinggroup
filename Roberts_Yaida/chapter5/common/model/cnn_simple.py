import torch
from torch import Tensor, nn

import logging

class CIFAR10ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope_positive = None
        self.slope_negative = None
        #self.do_dropout = False

        self.kernel_size_=(3, 3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=self.kernel_size_, padding='same')
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_size_, padding='same')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size_, padding='same')
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size_, padding='same')
        self.linear5 = nn.Linear(4096, 256)
        self.linear6 = nn.Linear(256, 10)
        self.max_pool1 = nn.MaxPool2d(kernel_size = (2,2))
        self.max_pool2 = nn.MaxPool2d(kernel_size = (2,2))
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)

        """
    def zero_grad(self):
        super().zero_grad()
        self.map1 = None
        self.map2 = None
        self.map3 = None        
        if self.map1 is not None and self.map1.grad is not None:
            self.map1.grad.detach_()
            self.map1.grad.zero_()
        if self.map2 is not None and self.map2.grad is not None:
            self.map2.grad.detach_()
            self.map2.grad.zero_()
        if self.map3 is not None and self.map3.grad is not None:
            self.map3.grad.detach_()
            self.map3.grad.zero_()
        """

    def set_slopes(self, slope_positive = 1.0, slope_negative = 0.25):
        self.slope_positive = slope_positive
        self.slope_negative = slope_negative

    def gen_dropout(self, input, prob):
        with torch.no_grad():
            return F.dropout(torch.ones_like(input), p=prob).detach().clone()
        #return torch.bernoulli(torch.full_like(input, prob))/(1-prob)

    def PReLU(self, input: Tensor) -> Tensor:
        input = torch.where(input >= 0, self.slope_positive * input, self.slope_negative * input)
        return input
    
    def forward_(self, x):
        return self.forward(x)
    
    def forward(self, x):
        #f_dropout = self.dropout
        #channels*size*size
        #3*32*32->16*32*32 <padding='same'?>
        h1 = self.PReLU(self.conv1(x))
        #16*32*32->32*32*32->32*16*16 <padding='same'?>
        h2 = self.max_pool1(self.PReLU(self.conv2(h1)))
        if self.training:
            h2 = self.dropout1(h2)
        #32*16*16->32*16*16 <padding='same'?>
        h3 = self.PReLU(self.conv3(h2))
        #32*16*16->64*16*16->64*8*8 <padding='same'?>
        h4 = self.max_pool2(self.PReLU(self.conv4(h3)))
        if self.training:
            h4 = self.dropout2(h4)
        #64*8*8->4096->256
        h4flat = torch.flatten(h4, 1)
        h5 = self.PReLU(self.linear5(h4flat))
        if self.training:
            h5 = self.dropout3(h5)
        return self.linear6(h5)
    
    def init_weights(self, cb=0.0, cw=1.0):
        logging.debug("CNN weights initialisation with cb={}, cw={}".format(cb, cw))

        #Weight initialisation as in 2.19, 2.20
        self.cb, self.cw = cb, cw
        kernel_size = self.kernel_size_[0]*self.kernel_size_[1]
        self.init_linear_weights(self.conv1, True, cb, cw/kernel_size)
        self.init_linear_weights(self.conv2, True, cb, cw/kernel_size)
        self.init_linear_weights(self.conv3, True, cb, cw/kernel_size)
        self.init_linear_weights(self.conv4, True, cb, cw/kernel_size)
        self.init_linear_weights(self.linear5, True, cb, cw/self.linear5.in_features)
        self.init_linear_weights(self.linear6, True, cb, cw/self.linear6.in_features)

    @staticmethod
    def init_linear_weights(linear, bias_on, var_b=0.0, var_w=1.0):
        nn.init.normal_(linear.weight, mean = 0., std = math.sqrt(var_w)) #approach via torch
        if bias_on:
            nn.init.normal_(linear.bias, mean = 0., std = math.sqrt(var_b))

    @staticmethod
    def init_linear_zeros(linear, bias_on):
        nn.init.zeros_(linear.weight)
        if bias_on:
            nn.init.zeros_(linear.bias)
