import types
import math
import torch
import torch.nn as nn
import numpy as np

class FeedForwardNet(nn.Module):

    def __init__(self, n0=3, nk=10, nl=3, l=3, bias_on=False):
        '''n0: # dimension of x
           nk: # hidden nodes
           nl: # dimension of y
           l: # number of layers
           bias_on: # whether bias is included into linear preactivations'''
        super().__init__()
#        if l < 2:
#            raise Exception("FFN must have at least two layers")
        self.n0=n0
        self.nk=nk
        self.nl=nl
        self.bias_on = bias_on
        self.log_level = None
        self.hidden_linears = []
        self.output_linear = None
        # assume layer-independent cb and cw
        self.cb, self.cw = None, None
        print("FeedForwardNet created with n0={}, nk={}, nl={}, l={}, bias_on={}".format(n0, nk, nl, l, bias_on))

        if l > 0:
            self.hidden_linears.append(nn.Linear(n0, nk, bias=bias_on))
        if l > 2:
            for _ in range(2, l):
                self.hidden_linears.append(nn.Linear(nk, nk, bias=bias_on))
        if l > 1:
            self.output_linear = nn.Linear(nk, nl, bias=bias_on)

    def set_log_level(self, value):
        self.log_level = value

    def get_log_level(self):
        if self.log_level in ("debug", "info", "warning", "error"):
            return self.log_level
        else:
            return "info"

    def init_weights(self, cb=0.0, cw=1.0):
        if self.get_log_level() == "debug":
            print("FeedForwardNet weights initialisation with cb={}, cw={}".format(cb, cw))

        #Weight initialisation as in 2.19, 2.20
        self.cb, self.cw = cb, cw
        #n_prev = self.n0
        for linear in self.hidden_linears:
            self.init_linear_weights(linear, self.bias_on, cb, cw/linear.in_features)
            #self.init_linear_weights(linear, self.bias_on, cb, cw/n_prev)
            #n_prev = linear.weight.size()[0]#???

        #self.init_linear_weights(self.output_linear, self.bias_on, cb, cw/n_prev)
        self.init_linear_weights(self.output_linear, self.bias_on, cb, cw/self.output_linear.in_features)

    @staticmethod
    def init_linear_weights(linear, bias_on, var_b=1.0, var_w=1.0):
        nn.init.normal_(linear.weight, mean = 0., std = math.sqrt(var_w)) #approach via torch
        '''
        rows, cols, dims = linear.out_features, linear.in_features, linear.out_features * linear.in_features
        data = np.reshape(np.random.normal(0, math.sqrt(var_w), dims), (rows,cols))
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(data).float())
        '''
        if bias_on:
            nn.init.normal_(linear.bias, mean = 0., std = math.sqrt(var_b))

class FFNOnForwardLogging(FeedForwardNet):
    def __init__(self, n0=3, nk=10, nl=3, l=3, bias_on=False):
        super().__init__(n0, nk, nl, l, bias_on)
        self.on_forward_step_activ = list() # callbacks for forward_steps_activation
        self.on_forward_step_preactiv = list() # callbacks for forward_steps_preactivation

    def register_on_forward_step_activ_callback(self, callback: types.FunctionType):
        self.on_forward_step_activ.append(callback)

    def register_on_forward_step_preactiv_callback(self, callback: types.FunctionType):
        self.on_forward_step_preactiv.append(callback)

    def trigger_on_forward_step_activ_callbacks(self, zk_):
        for callback in self.on_forward_step_activ:
            callback(zk_)

    def trigger_on_forward_step_preactiv_callbacks(self, zk_):
        for callback in self.on_forward_step_preactiv:
            callback(zk_)

class FFNGmetricLogging(FFNOnForwardLogging):
    def __init__(self, n0=3, nk=10, nl=3, l=3, bias_on=False):
        super().__init__(n0, nk, nl, l, bias_on)
        self.GXX = None  # record the flow of metric G (of type 4.8 and 4.36)
        self.g_indices = None # trainpoint-indices for the metric G flow to record
        self.register_on_forward_step_activ_callback(self.log_gmetric)
        self.PRE = None  # record the flow of preactivation
        self.pre_indices = None # trainpoint-indices for the preactivation flow to record
        self.register_on_forward_step_preactiv_callback(self.log_preactivation)        

    #pre_indices is an array of integers
    def set_preactivation_recording_indices(self, pre_indices):
        self.pre_indices = pre_indices
    
    def log_preactivation(self, zk_):
        if self.pre_indices != None:
            for key, values in self.PRE.items():
                if values == None:
                    values = []
                    self.PRE[key] = values
                #index = key
                values.append((zk_[key-1]))

    def get_preactivation(self, index):
        try:
            return self.PRE[index]
        except KeyError:
            if self.get_log_level() in ("debug", "info"):
                print("practivation for index {} not found".format(index))
            return []

    #g_indices is an array of tuples
    def set_gmetric_recording_indices(self, g_indices):
        self.g_indices = g_indices

    def log_gmetric(self, zk_):
        if self.g_indices != None:
            for key, values in self.GXX.items():
                if values == None:
                    values = []
                    self.GXX[key] = values
                (index_one, index_two) = key
                values.append(self.G_xx(zk_[index_one-1], zk_[index_two-1], self.cb, self.cw))

    def get_gmetric(self, index_one, index_two):
        try:
            return self.GXX[(index_one, index_two)]
        except KeyError:
            if self.get_log_level() in ("debug", "info"):
                print("gmetric for indices {} and {} not found".format(index_one, index_two))
            return []
    
    @staticmethod
    def G_xx(inbound_one, inbound_two, cbn, cwn):
        '''Calculation of metrics of type (4.8) and (4.36)'''
        return cbn + cwn*inbound_one.dot(inbound_two)/len(inbound_one)
