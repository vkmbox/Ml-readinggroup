import numpy as np
import math
from sklearn.metrics import mean_squared_error
from random import randrange

import torch
from torch import Tensor
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.func import functional_call, vmap, vjp, jvp, grad

from scipy.linalg import svdvals, norm
from scipy.special import softmax
from scipy.optimize import minimize_scalar

from common.util import *

import logging

def ntkvp2_np(func_single, func_mul, params, x1, x2, delta, lbd_dict=None):
    v = torch.from_numpy(np.transpose(delta))
    result = ntkvp2(func_single, func_mul, params, x1, x2, v, lbd_dict)
    return np.transpose(result.detach().numpy())

# Faster version of ntkvp. Computes sum_{j,b} H_{i,j,a,b} v_{j,b}. Contributed by Zhang Allan
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

class ParameterProcessor:
    def __init__(self):
        self.theta_current = {}
        self.delta_current = {}
        self.grad_current = {}

    def is_delta_empty(self):
        return len(self.delta_current) <= 0

    def save_theta(self, model):
        with torch.no_grad():        
            for name, param in model.named_parameters():
                self.theta_current[name] = param.detach().clone()

    def set_theta(self, model, momentum, eta, eta_scale = 1.0):
        with torch.no_grad():
            for name, param in model.named_parameters():
                grad = self.grad_current[name]
                delta = self.delta_current.get(name, None)
                if momentum <= 0.0 and eta_scale > 0.0:
                    param.data.copy_(self.theta_current[name] -eta * eta_scale * grad)
                elif eta_scale <= 0.0 and momentum > 0.0:
                    if delta is not None:
                        param.data.copy_(self.theta_current[name] + momentum * delta)
                    else:
                        param.data.copy_(self.theta_current[name])
                elif eta_scale > 0.0 and momentum > 0.0:
                    if delta is not None:
                        param.data.copy_(self.theta_current[name] + momentum * delta - eta * eta_scale * grad)
                    else:
                        param.data.copy_(self.theta_current[name] - eta * eta_scale * grad)
                else:
                    raise ValueError("eta_scale or momentum must be in interval(0., 1.)")

    def save_delta_current(self, momentum, eta, eta_scale = 1.0):
        for name, grad in self.grad_current.items():
            delta = self.delta_current.get(name, None)
            if delta is None or momentum <= 0.0:
                self.delta_current[name] = -eta * eta_scale * grad
            else:
                self.delta_current[name] = momentum * delta - eta * eta_scale * grad

    #Grad optionally multiplied by lambda
    def calc_autograd(self, model, loss, lambda_dict=None):
        param_buffer ={}
        for name, param in model.named_parameters():
            param_buffer[name] = param        
        df = torch.autograd.grad(loss, param_buffer.values(), retain_graph=True, create_graph=True, allow_unused=True)
        ii = 0
        for name in param_buffer:
            lambda_value = lambda_dict.get(name, 1.) if lambda_dict is not None else 1.
            self.grad_current[name] = lambda_value * df[ii].detach().clone()
            ii += 1
    
class OptimiserEtaSoftmaxArmihoNorm2Base:
    def __init__(self, meta, momentum=0.9, lbd_dict=None):
        self.meta = meta
        self.momentum = momentum
        self.step_one = 1.0
        #self.step_deltas = [0.0]
        self.step_deltas = [0.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625, -0.015625]
        #self.step_deltas = [3.0, -2.0, -1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625]
        #self.step_deltas = [1.0, -1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625]
        self.c1 = 0.0 #0.000000005 #0.000005 #0.000025
        self.c2 = 0.99 #0.75 #0.5#
        self.c3 = 0.99

        self.epsilon = 1e-8
        self.epsilon_criteria = 1e-5
        self.eta_min = self.epsilon
        self.lbd_dict = lbd_dict
        self.criterion = nn.CrossEntropyLoss()
        self.paramProcessor = ParameterProcessor()
        self.check_dropout = True

    def calc_eta(self, testNet, xx, pp, qq):
        meta = self.meta
        delta = pp - qq
        delta0 = reduce_to_active(delta, pp)

        logging.info("##Calculating reduced NTK")
        params = {k: v.detach().clone() for k, v in testNet.named_parameters()}
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
        return eta_ones, eta

    def step(self, testNet, labels, xx, momentum, Nesterov = False, use_ones = False):
        meta = self.meta
        #logging.info("##Bias_0 step-start:{}".format(testNet.conv1.bias[0].item()))

        self.paramProcessor.save_theta(testNet)
        if momentum > 0.0 and Nesterov == True and self.paramProcessor.is_delta_empty() == False:
            self.paramProcessor.set_theta(testNet, momentum, 0., 0.)
            #logging.info("##Bias_0 Nesterov pre-set:{}".format(testNet.conv1.bias[0].item()))

        if self.check_dropout:
            do_dropout_val = testNet.do_dropout
            testNet.do_dropout = False
        with torch.no_grad():
            logits_detached = testNet.forward_(xx).detach().numpy().copy()
        pp = labels_to_softhot(labels, meta.output_dim)
        qq = softmax(np.transpose(logits_detached), axis=(0)) + self.epsilon
        eta_ones, eta = self.calc_eta(testNet, xx, pp, qq)
        if self.check_dropout:
            testNet.do_dropout = do_dropout_val

        if use_ones:
            eta = eta_ones
        logging.info("##Eta value = {}".format(eta))
        if eta < self.eta_min or math.isnan(eta):
            logging.info("##Eta changed from {} to {}".format(eta, self.eta_min))
            eta = self.eta_min

        logging.info("##Calculating params-delta")
        testNet.zero_grad()
        logits = testNet.forward_(xx)
        loss = self.criterion(logits, labels)
        self.paramProcessor.calc_autograd(testNet, loss, self.lbd_dict)
        #logging.info("##Bias_0 calc_autograd:{}".format(self.paramProcessor.grad_current['conv1.bias'][0].item()))

        if self.check_dropout:
            do_dropout_val = testNet.do_dropout
            testNet.do_dropout = False
        eta_scale, eta, logits_k = self.backtrack_armiho_wolf_additional(testNet, xx, pp, qq, eta, momentum)
        if self.check_dropout:
            testNet.do_dropout = do_dropout_val
        self.paramProcessor.save_delta_current(momentum, eta, eta_scale)
        #logging.info("##Bias_0 delta:{}".format(self.paramProcessor.delta_current['conv1.bias'][0].item()))
        #logging.info("##Bias_0 step-finnish:{}".format(testNet.conv1.bias[0].item()))
        return eta_scale*eta, logits_k
    
    #Armiho: Loss(θ+α) <= c1*α*∇Loss(θ) + Loss(θ)
    #Wolf: |∇Loss(θ+α)| <= c2*|∇Loss(θ)|
    #0<c1<c2<1
    #Additional condition: ∇Loss(θ+α) <= c3*|∇Loss(θ)| , 0<c3<1 (Significant loss growth at the final point is unacceptable)
    def backtrack_armiho_wolf_additional(self, testNet, xx, pp, qq, eta, momentum):
        meta = self.meta
        qq_reduced = reduce_to_active(qq, pp)
        #initialLoss = -np.sum(pp * np.log(qq))/meta.batch_size
        initialLoss = -np.sum(np.log(qq_reduced))/meta.batch_size
        ratio = reduce_to_active(pp/qq, pp)
        logging.info("##Initial loss = {}".format(initialLoss))

        with torch.no_grad():
            #Initial 1.0 forward
            self.paramProcessor.set_theta(testNet, momentum, eta, self.step_one)
            #logging.info("##Bias_0 initial 1.0:{}".format(testNet.conv1.bias[0].item()))
            logits_one = testNet.forward_(xx)
            qq_one = softmax(np.transpose(logits_one.detach().numpy().copy()), axis=(0)) + self.epsilon
            delta_delta_one = reduce_to_active(qq - qq_one, pp) #We consider pp/qq without -

            logits_k = None
            eta_scale = self.step_one
            for step_delta in self.step_deltas:
                if (step_delta != 0.0):
                    eta_scale += step_delta
                    self.paramProcessor.set_theta(testNet, momentum, eta, eta_scale)
                    #logging.info("##Bias_0 adjustment to {}:{}".format(eta_scale, testNet.conv1.bias[0].item()))
                logits_k = testNet.forward_(xx)
                qq_k = softmax(np.transpose(logits_k.detach().numpy().copy()), axis=(0)) + self.epsilon
                qq_k_reduced = reduce_to_active(qq_k, pp)
                #loss_k = -np.sum(pp * np.log(qq_k))/meta.batch_size
                loss_k = -np.sum(np.log(qq_k_reduced))/meta.batch_size
                ratio_k = reduce_to_active(pp/qq_k, pp)
                #diff_est_k = np.dot(ratio_k, delta_delta_est) #∇L(θ+α)
                logging.info("##Step with eta_scale: {}, loss = {}".format(eta_scale, loss_k))
                logging.info("##Factual qq_k for ones: min={}, max={}, avg={}"\
                             .format(np.min(qq_k_reduced), np.max(qq_k_reduced), np.average(qq_k_reduced)))
                
                diff_fact = np.dot(ratio, delta_delta_one)
                diff_fact_k = np.dot(ratio_k, delta_delta_one)
                pq_ratio_k = 1.0
                try:
                    pq_ratio_k = norm(1 - qq_k_reduced, ord=2)/norm(1 - qq_reduced, ord=2)
                except ValueError as ex:
                    logging.error("Error when (p-q)ratio norm-2 for ones calculated: {}".format(ex))
                logging.info("##With factual qq_k: diff_init = {}, diff_k = {}, (p-q)ratio norm-2 for ones: {}"\
                            .format(diff_fact, diff_fact_k, pq_ratio_k))
                condition_armiho = loss_k - self.epsilon_criteria <= initialLoss + self.c1*eta_scale*diff_fact
                condition_wolf = abs(diff_fact_k) - self.epsilon_criteria <= self.c2*abs(diff_fact)
                condition_additional = diff_fact_k - self.epsilon_criteria <= self.c3*abs(diff_fact)
                logging.info("##Conditions: armiho = {}, wolf = {}, additional = {}, "\
                             .format(condition_armiho, condition_wolf, condition_additional))
                if condition_armiho: #and condition_wolf and condition_additional:
                    break

        return eta_scale, eta, logits_k