import math
import torch.nn as nn

class FeedForwardNet(nn.Module):

    def __init__(self, n0=3, nk=10, nl=3, l=3, bias_on=False):
        '''n0: # dimension of x
           nk: # hidden nodes
           nl: # dimension of y
           l: # number of layers
           bias_on: # whether bias is included into linear preactivations'''
        if l < 2:
            raise Exception("FFN must have at least two layers")
        super().__init__()
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

        self.hidden_linears.append(nn.Linear(n0, nk, bias=bias_on))
        if l > 2:
            for _ in range(2, l):
                self.hidden_linears.append(nn.Linear(nk, nk, bias=bias_on))
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
        n_prev = self.n0
        for linear in self.hidden_linears:
            self.init_linear_weights(linear, self.bias_on, math.sqrt(cb), math.sqrt(cw/n_prev))
            n_prev = linear.weight.size()[0]

        self.init_linear_weights(self.output_linear, self.bias_on, math.sqrt(cb), math.sqrt(cw/n_prev))

    @staticmethod
    def init_linear_weights(linear, bias_on, std_b=1.0, std_w=1.0):
        nn.init.normal_(linear.weight, mean = 0., std = std_w)
        n_prev = linear.weight.size()[0]
        if bias_on:
            nn.init.normal_(linear.bias, mean = 0., std = std_b)

class FFNGmetricLogging(FeedForwardNet):
    def __init__(self, n0=3, nk=10, nl=3, l=3, bias_on=False):
        super().__init__(n0, nk, nl, l, bias_on)
        self.GXX = None  # record the flow of metric G (of type 4.8 and 4.36)
        self.g_indices = None # trainpoint-indices for the flow to record

    #g_indices is an array of tuples of size number_of_index_pairs_to_track
    def set_gmetric_recording_indices(self, g_indices):
        self.g_indices = g_indices

    def log_gmetric(self, zk):
        zk_ = zk.detach().numpy()
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
