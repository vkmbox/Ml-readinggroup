import torch
from torch import nn
from torch.func import vmap, vjp, jvp, grad, functional_call
import numpy as np
from scipy.linalg import norm
from scipy.special import softmax
from scipy.optimize import minimize_scalar

from common.util import labels_to_softhot

import logging

# Faster version of ntkvp. Computes sum_{j,b} H_{i,j,a,b} v_{j,b}. Contributed by Zhang Allan
def ntkvp2_np(func_single, func_mul, params, x1, x2, delta, lbd_dict=None):
    v = torch.from_numpy(np.transpose(delta))
    result = ntkvp2(func_single, func_mul, params, x1, x2, v, lbd_dict)
    return np.transpose(result.detach().numpy())

def ntkvp2(func_single, func_mul, params, x1, x2, v, lbd_dict=None):
    '''
    lbd_dict: dict ~ {param_name: lambda},
        if None, all lambda = 1
    v ~ (n_samples * output_dim)
    x1, x2 ~ (n_samples * input_dim)
    '''
    vjps = grad(lambda pa: (func_mul(pa, x2)*v).sum())(params)
    if lbd_dict is not None:
        for pn in vjps:
            vjps[pn] *= lbd_dict.get(pn, 1.)
    vjps = (vjps,)
    def get_ntkv(x1, vjps):
        def func_x1(params):
                return func_single(params, x1)
        # This computes J(X1) @ vjps
        _, jvps = jvp(func_x1, (params,), vjps)
        return jvps

    result = vmap(get_ntkv, (0, None))(x1, vjps)
    return result

def solve_eta_norm2(delta_r, delta):
    '''
    get eta which minimizes rhs of \inf.86
    delta_r, delta ~ (n_samples, n_outputs)
    delta: p - q
    delta_r: R_ab
    '''
    return np.sum((delta*delta_r)) / np.sum((delta_r**2))

def NTK_softmaxV3(HL, qq, meta):
    SM = (np.identity(meta.output_dim)[:,:,None] - qq[:,None,:])*qq[None,:,:]
    return np.sum(SM[:,:,:]*HL[:,None,:], axis=0)

def reduce_to_active(MX_FULL, pp):
    return np.sum(MX_FULL*pp, axis=0)

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
            testNet.input_fc.weight += self.delta_weight_00 * delta
            testNet.input_fc.bias += self.delta_bias_00 * delta
            testNet.hidden_fc.weight += self.delta_weight_01 * delta
            testNet.hidden_fc.bias += self.delta_bias_01 * delta
            testNet.output_fc.weight += self.delta_weight_02 * delta
            testNet.output_fc.bias += self.delta_bias_02 * delta

class StepCalculatorDeltas(StepCalculatorBase):
    def __init__(self, meta):
        super().__init__()
        self.meta = meta
        self.step_one = 1.0
        self.step_deltas = [0.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625, -0.015625]
        #self.step_deltas = [3.0, -2.0, -1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625]
        #self.step_deltas = [1.0, -1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625]
        self.c1 = 0.0 #0.000000005 #0.000005 #0.000025
        self.c2 = 0.99 #0.75 #0.5#
        self.c3 = 0.25

    def calculate_deltas(self, testNet, loss, eta, lbd_dict=None):
        meta = self.meta
        df = torch.autograd.grad(loss, (testNet.input_fc.weight, testNet.input_fc.bias\
                                        , testNet.hidden_fc.weight, testNet.hidden_fc.bias\
                                        , testNet.output_fc.weight, testNet.output_fc.bias)\
            , retain_graph=True, create_graph=True, allow_unused=True)
        self.delta_weight_00 = -eta * (lbd_dict.get('input_fc.weight', 1.) if lbd_dict is not None else 1.) * df[0].detach()
        self.delta_bias_00 = -eta * (lbd_dict.get('input_fc.bias', 1.) if lbd_dict is not None else 1.) *df[1].detach()
        self.delta_weight_01 = -eta * (lbd_dict.get('hidden_fc.weight', 1.) if lbd_dict is not None else 1.) *df[2].detach()
        self.delta_bias_01 = -eta * (lbd_dict.get('hidden_fc.bias', 1.) if lbd_dict is not None else 1.) *df[3].detach()
        self.delta_weight_02 = -eta * (lbd_dict.get('output_fc.weight', 1.) if lbd_dict is not None else 1.) *df[4].detach()
        self.delta_bias_02 = -eta * (lbd_dict.get('output_fc.bias', 1.) if lbd_dict is not None else 1.) *df[5].detach()

    #Armiho: Loss(θ+α) <= c1*α*∇Loss(θ) + Loss(θ)
    #Wolf: |∇Loss(θ+α)| <= c2*|∇Loss(θ)|
    #0<c1<c2<1
    #Additional condition: ∇Loss(θ+α) <= c3*|∇Loss(θ)| , 0<c3<1 (Significant loss growth at the final point is unacceptable)
    def backtrack_armiho_wolf_additional(self, testNet, xx, pp, qq, eta):
        meta = self.meta
        initialLoss = -np.sum(pp * np.log(qq))/meta.batch_size
        ratio = reduce_to_active(pp/qq, pp)
        logging.info("##Initial loss = {}".format(initialLoss))

        with torch.no_grad():
            self.do_step0(testNet, self.step_one)
            logits_one = testNet.forward_(xx)
            qq_one = softmax(np.transpose(logits_one.detach().numpy()), axis=(0))
            delta_delta_one = reduce_to_active(qq - qq_one, pp) #We consider pp/qq without -

        logits_k = None
        aa = self.step_one
        with torch.no_grad():
            for step_delta in self.step_deltas:
                if (step_delta != 0.0):
                    aa += step_delta
                    self.do_step0(testNet, step_delta)
                logits_k = testNet.forward_(xx)
                qq_k = softmax(np.transpose(logits_k.detach().numpy()), axis=(0))
                #logging.info("qq1 factual: {}".format(reduce_to_active(qq_k, pp)))
                loss_k = -np.sum(pp * np.log(qq_k))/meta.batch_size
                ratio_k = reduce_to_active(pp/qq_k, pp)
                #diff_est_k = np.dot(ratio_k, delta_delta_est) #∇L(θ+α)
                logging.info("##Step with aa: {}, loss_k: {}".format(aa, loss_k))
                qq_k_reduced = reduce_to_active(qq_k, pp)
                logging.info("##Factual qq_k for ones: min={}, max={}, avg={}"\
                             .format(np.min(qq_k_reduced), np.max(qq_k_reduced), np.average(qq_k_reduced)))
                
                diff_fact = np.dot(ratio, delta_delta_one)
                diff_fact_k = np.dot(ratio_k, delta_delta_one)
                pq_ratio_k = norm(pp - qq_k, ord=2)/norm(pp - qq, ord=2)
                logging.info("##With factual qq_k: diff_init = {}, diff_k = {}, (p-q)ratio norm-2: {}"\
                            .format(diff_fact, diff_fact_k, pq_ratio_k))
                condition_armiho = loss_k <= initialLoss + self.c1*aa*diff_fact
                condition_wolf = abs(diff_fact_k) <= self.c2*abs(diff_fact)
                condition_additional = diff_fact_k <= self.c3*abs(diff_fact)
                logging.info("##Conditions: armiho = {}, wolf = {}, additional = {}, "\
                             .format(condition_armiho, condition_wolf, condition_additional))
                if condition_armiho and condition_wolf and condition_additional:
                    break

        return eta*aa, logits_k

class StepCalculatorEtaSoftmaxArmihoNorm2Base(StepCalculatorDeltas):
    def __init__(self, meta, lbd_dict=None):
        super().__init__(meta)
        self.lbd_dict = lbd_dict
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        #self.criterion1 = nn.Softmax(dim=1)

    def calc_eta(self, testNet, xx, pp, qq): #, delta, delta0
        meta = self.meta
        delta = pp - qq
        delta0 = reduce_to_active(delta, pp)

        logging.info("##Calculating reduced NTK")
        params = {k: v.detach() for k, v in testNet.named_parameters()}
        fnet_single = lambda params, x: functional_call(testNet, params, (x.unsqueeze(0),)).squeeze(0)
        fnet_mul = lambda params, x: functional_call(testNet, params, (x,))
        #Reduced NTK
        #delta_r = ntkvp_np(fnet_single, params, xx, xx, delta, self.lbd_dict)
        delta_r = ntkvp2_np(fnet_single, fnet_mul, params, xx, xx, delta, self.lbd_dict)

        logging.info("##Calculating step-forward and eta")
        NL_r = NTK_softmaxV3(delta_r, qq, meta)
        delta0_r = reduce_to_active(NL_r, pp)
        eta = solve_eta_norm2(NL_r, delta) #for all n,α
        eta_ones = solve_eta_norm2(delta0_r, delta0) #for each α only n with pp=1 is taken
        logging.info("##Calculated eta for ones = {}, general eta = {}".format(eta_ones, eta))
        return eta_ones

    def step(self, testNet, logits_detached, labels, xx, eta_top = 0.0):
        meta = self.meta

        pp = labels_to_softhot(labels, meta.output_dim)
        qq = softmax(np.transpose(logits_detached), axis=(0))

        eta = self.calc_eta(testNet, xx, pp, qq)
        if eta_top > 0.0 and eta > eta_top:
            eta = eta_top
            logging.info("##Smoothing eta, result eta={}".format(eta))

        logging.info("##Calculating params-delta")
        testNet.zero_grad()
        logits = testNet.forward_(xx)
        loss = self.criterion(logits, labels)
        self.calculate_deltas(testNet, loss, eta, self.lbd_dict)

        return self.backtrack_armiho_wolf_additional(testNet, xx, pp, qq, eta)
    
##Step calculates η minimising (∞.86) softmax-analogue with 2-matrix norm. 
## Parameters change is according to (7.11) for cross-enthropy loss, step is adjusted by Armiho/Wolf condition
class StepCalculatorEtaSoftmaxArmihoNorm2(StepCalculatorEtaSoftmaxArmihoNorm2Base):
    def __init__(self, meta):
        lbd_dict = {}
        lbd_dict['input_fc.weight'] = meta.lw_input()
        lbd_dict['hidden_fc.weight'] = meta.lw_hidden()
        lbd_dict['output_fc.weight'] = meta.lw_output()
        lbd_dict['input_fc.bias'] = meta.lb
        lbd_dict['hidden_fc.bias'] = meta.lb
        lbd_dict['output_fc.bias'] = meta.lb
        super().__init__(meta, lbd_dict)

##Step calculates η minimising (∞.86) softmax-analogue with 1-matrix norm. 
## Parameters change is according to (7.11) for cross-enthropy loss, step is adjusted by Armiho/Wolf condition
class StepCalculatorEtaSoftmaxArmihoNorm1Base(StepCalculatorDeltas):
    def __init__(self, meta, lbd_dict=None):
        super().__init__(meta)
        self.lbd_dict = lbd_dict
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def step(self, testNet, logits_detached, labels, xx): #, logits
        meta = self.meta

        pp = labels_to_softhot(labels, meta.output_dim)
        qq = softmax(np.transpose(logits_detached), axis=(0))
        delta = pp - qq

        logging.info("##Calculating reduced NTK and eta")
        params = {k: v.detach() for k, v in testNet.named_parameters()}
        fnet_single = lambda params, x: functional_call(testNet, params, (x.unsqueeze(0),)).squeeze(0)
        fnet_mul = lambda params, x: functional_call(testNet, params, (x,))
        #Reduced NTK
        #delta_r = ntkvp_np(fnet_single, params, xx, xx, delta, self.lbd_dict)
        delta_r = ntkvp2_np(fnet_single, fnet_mul, params, xx, xx, delta, self.lbd_dict)

        logging.info("##Calculating step-forward and eta")
        #testNet.zero_grad()
        #logits = testNet.forward_(xx)
        NL_r = NTK_softmaxV3(delta_r, qq, meta)
        logging.info("##Calculating eta minimising norm_1 for delta_0 - eta*R")
        delta_flat, NL_r_flat = delta.flatten(), NL_r.flatten()
        fun = lambda eta: norm(delta_flat - eta*NL_r_flat, ord=1)
        res = minimize_scalar(fun, bounds=(0, 1000))
        logging.info("##Optimal point for all found: {}".format(res))
        eta = res.x
        #for ones in softmax-encoding
        delta_ones, NL_r_ones = reduce_to_active(delta, pp), reduce_to_active(NL_r, pp)
        fun_ones = lambda eta_ones: norm(delta_ones - eta_ones*NL_r_ones, ord=1)
        res_ones = minimize_scalar(fun_ones, bounds=(0, 1000))
        logging.info("##Optimal point for ones found: {}".format(res_ones))
        eta_ones = res_ones.x        
        logging.info("##Calculated eta = {}, eta_ones = {}".format(eta, eta_ones))
        logging.info("##Calculating params-delta")
        testNet.zero_grad()
        logits = testNet.forward_(xx)
        loss = self.criterion(logits, labels)
        self.calculate_deltas(testNet, loss, eta_ones, self.lbd_dict) #Set eta otherwise!!!

        return self.backtrack_armiho_wolf_additional(testNet, xx, pp, qq, eta_ones) #Set eta otherwise!!!

##Step calculates η minimising (∞.86) softmax-analogue with 1-matrix norm. 
## Parameters change is according to (7.11) for cross-enthropy loss, step is adjusted by Armiho/Wolf condition
class StepCalculatorEtaSoftmaxArmihoNorm1(StepCalculatorEtaSoftmaxArmihoNorm1Base):
    def __init__(self, meta):
        lbd_dict = {}
        lbd_dict['input_fc.weight'] = meta.lw_input()
        lbd_dict['hidden_fc.weight'] = meta.lw_hidden()
        lbd_dict['output_fc.weight'] = meta.lw_output()
        lbd_dict['input_fc.bias'] = meta.lb
        lbd_dict['hidden_fc.bias'] = meta.lb
        lbd_dict['output_fc.bias'] = meta.lb   
        super().__init__(meta, lbd_dict)

##Step calculates η from (7.12) privided Loss(t+1)=0 and λ_μν=I; criticality for NTK is ignored. 
##Parameters change is according to (7.11) with the criticality-calculating λ_μν
class StepCalculatorZhangLambda(StepCalculatorDeltas):
    def __init__(self, meta):
        super().__init__(meta)
        self.meta = meta
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def calculate_deltasZh1(self, testNet, eta):
        meta = self.meta
        self.delta_weight_00 = -eta * meta.lw_input() * testNet.input_fc.weight.grad.detach()
        self.delta_bias_00 = -eta * meta.lb * testNet.input_fc.bias.grad.detach()
        self.delta_weight_01 = -eta * meta.lw_hidden() * testNet.hidden_fc.weight.grad.detach()
        self.delta_bias_01 = -eta * meta.lb * testNet.hidden_fc.bias.grad.detach()
        self.delta_weight_02 = -eta * meta.lw_output() * testNet.output_fc.weight.grad.detach()
        self.delta_bias_02 = -eta * meta.lb * testNet.output_fc.bias.grad.detach()

    def calc_eta(self, testNet, loss_value):
        meta = self.meta
        # --calculate the factor--
        denominator = 0.0
        denominator += ((testNet.input_fc.weight.grad)**2).sum().item() * meta.lw_input()
        denominator += ((testNet.input_fc.bias.grad)**2).sum().item() * meta.lb
        denominator += ((testNet.hidden_fc.weight.grad)**2).sum().item() * meta.lw_hidden()
        denominator += ((testNet.hidden_fc.bias.grad)**2).sum().item() * meta.lb
        denominator += ((testNet.output_fc.weight.grad)**2).sum().item() * meta.lw_output()
        denominator += ((testNet.output_fc.bias.grad)**2).sum().item() * meta.lb
        #for param in testNet.parameters():
        #    denominator += ((param.grad)**2).sum().item()
        eta = loss_value/denominator
        logging.info("##Calculated eta = {}".format(eta))
        return eta

    def step(self, testNet, logits_detached, labels, xx):
        meta = self.meta

        logging.info("##Calculating step-forward and eta")
        testNet.zero_grad()
        logits = testNet.forward_(xx)
        loss = self.criterion(logits, labels)
        loss.backward()

        # --calculate the factor--
        eta = self.calc_eta(testNet, loss.item())
        logging.info("##Calculating params-delta")
        self.calculate_deltasZh1(testNet, eta)

        pp = labels_to_softhot(labels, meta.output_dim)
        qq = softmax(np.transpose(logits_detached), axis=(0))
        return self.backtrack_armiho_wolf_additional(testNet, xx, pp, qq, eta)

##Step calculates η from (7.12) privided Loss(t+1)=0 and λ_μν=I; criticality for NTK is ignored. 
##Parameters change is according to (7.11) with λ_μν=I
class StepCalculatorZhangSimplified(StepCalculatorDeltas):
    def __init__(self, meta):
        super().__init__(meta)
        self.meta = meta
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def calculate_deltasZhEta(self, testNet, eta):
        self.delta_weight_00 = -eta * testNet.input_fc.weight.grad.detach()
        self.delta_bias_00 = -eta * testNet.input_fc.bias.grad.detach()
        self.delta_weight_01 = -eta * testNet.hidden_fc.weight.grad.detach()
        self.delta_bias_01 = -eta * testNet.hidden_fc.bias.grad.detach()
        self.delta_weight_02 = -eta * testNet.output_fc.weight.grad.detach()
        self.delta_bias_02 = -eta * testNet.output_fc.bias.grad.detach()

    def calc_eta(self, testNet, loss_value):
        meta = self.meta
        denominator = 0.0
        denominator += ((testNet.input_fc.weight.grad)**2).sum().item()
        denominator += ((testNet.input_fc.bias.grad)**2).sum().item()
        denominator += ((testNet.hidden_fc.weight.grad)**2).sum().item()
        denominator += ((testNet.hidden_fc.bias.grad)**2).sum().item()
        denominator += ((testNet.output_fc.weight.grad)**2).sum().item()
        denominator += ((testNet.output_fc.bias.grad)**2).sum().item()
        #for param in testNet.parameters():
        #    denominator += ((param.grad)**2).sum().item()
        eta = loss_value/denominator
        logging.info("##Calculated eta = {}".format(eta))
        return eta

    def step(self, testNet, logits_detached, labels, xx):
        meta = self.meta

        logging.info("##Calculating step-forward and eta")
        testNet.zero_grad()
        logits = testNet.forward_(xx)
        loss = self.criterion(logits, labels)
        loss.backward()

        # --calculate the factor--
        eta = self.calc_eta(testNet, loss.item())

        logging.info("##Calculating params-delta")        
        self.calculate_deltasZhEta(testNet, eta)

        pp = labels_to_softhot(labels, meta.output_dim)
        qq = softmax(np.transpose(logits_detached), axis=(0))
        return self.backtrack_armiho_wolf_additional(testNet, xx, pp, qq, eta)
