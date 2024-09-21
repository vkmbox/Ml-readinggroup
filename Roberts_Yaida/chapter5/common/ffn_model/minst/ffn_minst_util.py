import numpy as np
import torch
import itertools as itr
from random import randrange
from scipy.linalg import norm
from scipy.optimize import minimize_scalar

import logging

import sys
if '../common' not in sys.path:
    sys.path.append('../common')
from common.util import labels_to_onehot

class MetaData:
    def __init__(self, batch_size = 96, input_dim = 784, input_width = 250, hidden_width = 100, output_dim = 26\
                 , lb = 1e-2, lw = 7.5):
        self.batch_size = batch_size
        self.input_dim = input_dim  # image 28*28
        self.input_width = input_width
        self.hidden_width = hidden_width
        self.output_dim = output_dim  # num of classes
        self.lb = lb
        self.lw = lw

    #lambdas_w as from (8.5)
    def lw_input(self):
        return self.lw/self.input_dim,
    
    def lw_hidden(self):
        return self.lw/self.input_width

    def lw_output(self):
        return self.lw/self.hidden_width

class NTKCalculator:
    def __init__(self, meta):
        self.meta = meta
        self.input_dweight = np.zeros((meta.output_dim, meta.batch_size, meta.input_width, meta.input_dim))
        self.input_dbias = np.zeros((meta.output_dim, meta.batch_size, meta.input_width))
        self.hidden_dweight = np.zeros((meta.output_dim, meta.batch_size, meta.hidden_width, meta.input_width))
        self.hidden_dbias = np.zeros((meta.output_dim, meta.batch_size, meta.hidden_width))
        self.output_dweight = np.zeros((meta.output_dim, meta.batch_size, meta.output_dim, meta.hidden_width))
        self.output_dbias = np.zeros((meta.output_dim, meta.batch_size, meta.output_dim))

    def calculate_derivatives(self, testNet, logits):
        for kk, alpha in itr.product(range(self.meta.output_dim), range(self.meta.batch_size)):
            df = torch.autograd.grad(logits[alpha,kk], (testNet.input_fc.weight, testNet.input_fc.bias\
                                            , testNet.hidden_fc.weight, testNet.hidden_fc.bias\
                                            , testNet.output_fc.weight, testNet.output_fc.bias)\
                , retain_graph=True, create_graph=True, allow_unused=True)
            self.input_dweight[kk, alpha] = df[0].detach().numpy()
            self.input_dbias[kk, alpha] = df[1].detach().numpy()
            self.hidden_dweight[kk, alpha] = df[2].detach().numpy()
            self.hidden_dbias[kk, alpha] = df[3].detach().numpy()
            self.output_dweight[kk, alpha] = df[4].detach().numpy()
            self.output_dbias[kk, alpha] = df[5].detach().numpy()

    def calculate_average_NTK(self, HL):
        meta = self.meta
        HL_avg = np.zeros((meta.batch_size, meta.batch_size))
        for alpha1, alpha2 in itr.product(range(meta.batch_size), range(meta.batch_size)):
            value_avg = np.average([HL[num, num, alpha1, alpha2] for num in np.arange(meta.output_dim)])
            HL_avg[alpha1, alpha2] = value_avg

        return HL_avg

    def calculate_NTK(self):
        meta = self.meta
        input_dbias_sum = meta.lb * np.tensordot(self.input_dbias, self.input_dbias, axes=([2],[2]))
        input_dweight_sum = meta.lw_input() * np.tensordot(self.input_dweight, self.input_dweight, axes=([2,3],[2,3]))
        hidden_dbias_sum = meta.lb * np.tensordot(self.hidden_dbias, self.hidden_dbias, axes=([2],[2]))
        hidden_dweight_sum = meta.lw_hidden() * np.tensordot(self.hidden_dweight, self.hidden_dweight, axes=([2,3],[2,3]))        
        output_dbias_sum = meta.lb * np.tensordot(self.output_dbias, self.output_dbias, axes=([2],[2]))
        output_dweight_sum = meta.lw_output() * np.tensordot(self.output_dweight, self.output_dweight, axes=([2,3],[2,3]))        
        summary = input_dbias_sum+input_dweight_sum+hidden_dbias_sum+hidden_dweight_sum+output_dbias_sum+output_dweight_sum
        return np.moveaxis(summary, 1, 2)


class StepCalculatorBase:
    def __init__(self):
        self.delta_weight_00 = None
        self.delta_bias_00 = None
        self.delta_weight_01 = None
        self.delta_bias_01 = None
        self.delta_weight_02 = None
        self.delta_bias_02 = None

    def do_step0(self, testNet, delta = 1.0):
        with torch.no_grad():
            testNet.input_fc.weight += torch.from_numpy(self.delta_weight_00) * delta
            testNet.input_fc.bias += torch.from_numpy(self.delta_bias_00) * delta
            testNet.hidden_fc.weight += torch.from_numpy(self.delta_weight_01) * delta
            testNet.hidden_fc.bias += torch.from_numpy(self.delta_bias_01) * delta
            testNet.output_fc.weight += torch.from_numpy(self.delta_weight_02) * delta
            testNet.output_fc.bias += torch.from_numpy(self.delta_bias_02) * delta

##Step calculates η minimising (∞.86) with 2-matrix norm. Parameters change is accordinng to (7.11)
class StepCalculatorEtaNtk(StepCalculatorBase):
    def __init__(self, meta, ntk_recalc_steps_period = 5):
        super().__init__()
        self.meta = meta
        self.ntk_recalc_steps_period = ntk_recalc_steps_period

    def step(self, testNet, labels, logits, etas):
        meta = self.meta

        calc0 = NTKCalculator(meta)
        calc0.calculate_derivatives(testNet, logits)
        do_calc_eta = (etas[0] == 0.0 or self.ntk_recalc_steps_period <= 0)
        if (do_calc_eta == False):
            do_calc_eta = randrange(self.ntk_recalc_steps_period) == 0

        HL = None
        eta = 0.0
        if (do_calc_eta):
            logging.info("##Calculating NTK by (8.4)")
            HL = calc0.calculate_NTK()
            HL_avg = calc0.calculate_average_NTK(HL)
            ID = np.identity(meta.batch_size)
            logging.info("##Calculating η minimising (∞.86): HL_avg = {}".format(HL_avg))
            fun = lambda eta: norm(ID - eta*HL_avg, ord=2) #np.inf)
            res = minimize_scalar(fun, bounds=(0, 1000))
            logging.info("##Optimal point found: {}".format(res))
            if etas[0] == 0.0:
                etas[1] = res.x
            else:
                etas[1] = etas[0]
            etas[0] = res.x
            eta = res.x
        else:
            eta = round(0.6 * etas[0] + 0.4 * etas[1], 5);
            logging.info("##Using previous eta value={}".format(eta))
        #(7.11)
        logits_detached = logits.detach()
        yy = labels_to_onehot(logits_detached, labels)

        #print("Calculating deltas")
        zz = np.transpose(logits_detached.numpy())
        term = eta*(yy - zz)
        self.delta_weight_00 = np.tensordot(term, calc0.input_dweight, axes=([0,1],[0,1])) * meta.lw_input()
        self.delta_bias_00 = np.tensordot(term, calc0.input_dbias, axes=([0,1],[0,1])) * meta.lb
        self.delta_weight_01 = np.tensordot(term, calc0.hidden_dweight, axes=([0,1],[0,1])) * meta.lw_hidden()
        self.delta_bias_01 = np.tensordot(term, calc0.hidden_dbias, axes=([0,1],[0,1])) * meta.lb
        self.delta_weight_02 = np.tensordot(term, calc0.output_dweight, axes=([0,1],[0,1])) * meta.lw_output()
        self.delta_bias_02 = np.tensordot(term, calc0.output_dbias, axes=([0,1],[0,1])) * meta.lb
        
        self.do_step0(testNet)
        return etas, HL
