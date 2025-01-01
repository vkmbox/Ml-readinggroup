import sys
if '../common' not in sys.path:
    sys.path.append('../common')

class MetaData:
    def __init__(self, batch_size = 96, input_dim = 784, input_width = 250, hidden_width = 100, output_dim = 26, reduction='mean'\
                 , lb = 1e-2, lw = 7.5, device='cpu', check_dropout=False):
        self.batch_size = batch_size
        self.input_dim = input_dim  # image 28*28
        self.input_width = input_width
        self.hidden_width = hidden_width
        self.output_dim = output_dim  # num of classes
        self.lb = lb
        self.lw = lw
        self.reduction=reduction
        self.device=device
        self.check_dropout=check_dropout

    #lambdas_w as from (8.5)
    def lw_input(self):
        return self.lw/self.input_dim
    
    def lw_hidden(self):
        return self.lw/self.input_width

    def lw_output(self):
        return self.lw/self.hidden_width
