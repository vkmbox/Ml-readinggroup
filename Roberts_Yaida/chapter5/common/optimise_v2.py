import numpy as np
import math
import itertools as itr

import torch
from torch import Tensor
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.func import functional_call, vmap, vjp, jvp, grad

from scipy.linalg import svdvals, norm
from scipy.special import softmax
from scipy.optimize import minimize_scalar

#from common.util import *

import logging

def calc_ce_loss(pp, qq, meta):
    loss = -torch.sum(pp * torch.log(qq))
    if meta.reduction == 'mean':
        loss = loss/meta.batch_size
    return loss

def labels_to_softhot(true_labels, meta):
    batch_size = true_labels.shape[0]
    with torch.no_grad():
        yy_softhot = torch.zeros(meta.output_dim, batch_size).to(meta.device)
        for batch_num in range(batch_size):
            yy_softhot[true_labels[batch_num], batch_num] = 1.0

    #logging.debug("For labels\n{}\nonehots are:\n{}".format(true_labels, yy_softhot))
    return yy_softhot

#Grad optionally multiplied by lambda
def calc_autograd(model, logits, meta):
    batch_size = logits.shape[0]
    param_buffer, grad_buffer ={}, {}
    for name, param in model.named_parameters():
        param_buffer[name] = param
        dimensions = list(param.shape)
        dimensions.insert(0, batch_size)
        dimensions.insert(0, meta.output_dim)
        grad_buffer[name] = torch.empty(dimensions)

    for alpha, kk in itr.product(range(batch_size), range(meta.output_dim)):
        df = torch.autograd.grad(logits[alpha,kk], param_buffer.values(), retain_graph=True, create_graph=True, allow_unused=True)
        with torch.no_grad():
            ii = 0
            for name in param_buffer:
                #update = df[ii].detach().clone()
                #print("update dim={}, buffer dim={}".format(update.shape, grad_buffer[name].shape))
                grad_buffer[name][kk,alpha] = df[ii].detach().clone()
                ii += 1

    return grad_buffer

def ntk_reduced(grad_buffer, delta, meta, lambda_dict=None):
    batch_size = delta.shape[1]
    with torch.no_grad():
        gamma2 = torch.zeros(meta.output_dim, batch_size).to(meta.device)
        for name, grad in grad_buffer.items():
            lambda_value = lambda_dict.get(name, 1.) if lambda_dict is not None else 1.
            grad_flatten = torch.flatten(grad, start_dim=2).to(meta.device) #TODO: check if possible to flattern via view
            term0 = torch.sum(grad_flatten*delta[:, :, None], (0,1)).to(meta.device)
            gamma2 += lambda_value*torch.sum(term0[None, None, :]*grad_flatten, (2))
        return gamma2

def ntk_softmax_reduced(gamma2, qq, meta):
    with torch.no_grad():
        SM = (torch.eye(meta.output_dim).to(meta.device)[:,:,None] - qq[:,None,:])*qq[None,:,:]
        return torch.sum(SM[:,:,:]*gamma2[:,None,:], 0)

def solve_eta_norm2(RR, delta): #RR, delta
    '''
    get eta which minimizes rhs of \inf.86
    delta_r, delta ~ (n_samples, n_outputs)
    delta: p - q
    delta_r: R_ab
    '''
    with torch.no_grad():
        return torch.sum(RR*delta) / torch.sum(RR**2)
    #eta1 = (torch.sum(pq_delta*qq_delta)/torch.sum(qq_delta*qq_delta)).item()

def reduce_to_active(matrix_full, pp):
    with torch.no_grad():
        return torch.sum(matrix_full*pp, 0)

class ParameterProcessor:
    def __init__(self, meta):
        self.theta_current = {}
        self.delta_current = {}
        self.grad_current = {}
        self.meta = meta

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
        with torch.no_grad():
            for name, grad in self.grad_current.items():
                delta = self.delta_current.get(name, None)
                if delta is None or momentum <= 0.0:
                    self.delta_current[name] = -eta * eta_scale * grad
                else:
                    self.delta_current[name] = momentum * delta - eta * eta_scale * grad

    #Grad optionally multiplied by lambda
    def calc_ce_theta_delta(self, grad_buffer, pp, qq, lambda_dict=None):
        meta = self.meta
        self.grad_current ={}
        with torch.no_grad():
            delta = (qq-pp).to(meta.device)
            grad_norm2_squared = 0.
            ii = 0
            for name, grad_pre0 in grad_buffer.items():
                grad_pre = grad_pre0.to(meta.device) #torch.flatten(grad_pre0, start_dim=0, end_dim=1).to(meta.device)
                lambda_value = lambda_dict.get(name, 1.) if lambda_dict is not None else 1.
                #if meta.reduction == 'mean':
                #    lambda_value = lambda_value/meta.batch_size
                grad = None
                if grad_pre.ndim == 3:
                    grad = torch.sum(grad_pre * delta[:,:,None], (0,1))
                    self.grad_current[name] = lambda_value * grad
                elif grad_pre.ndim == 4:
                    grad = torch.sum(grad_pre * delta[:,:,None, None], (0,1))
                    self.grad_current[name] = lambda_value * grad
                elif grad_pre.ndim == 5:
                    grad = torch.sum(grad_pre * delta[:,:,None, None, None], (0,1))
                    self.grad_current[name] = lambda_value * grad
                elif grad_pre.ndim == 6:
                    grad = torch.sum(grad_pre * delta[:,:,None, None, None, None], (0,1))
                    self.grad_current[name] = lambda_value * grad
                else:
                    raise Exception("Unexpected dim number for shape={}".format(grad_pre.shape))

                self.grad_current[name] = lambda_value * grad
                ii += 1
                grad_norm2_squared += ((grad)**2).sum().item() #TODO: meta.reduction == 'mean'

            return grad_norm2_squared

class StepProcessor:
    def __init__(self, meta, epsilon):
        self.meta=meta
        self.paramProcessor = ParameterProcessor(meta)
        self.epsilon = epsilon
        self.epsilon_criteria = 1e-5
        self.step_one = 1.0
        #self.step_deltas = [0.0]
        self.step_armiho = [0.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625] #, -0.015625]
        self.step_wolf = [1.0, 2.0, 4.0, 8.0, 16.0]
        #self.step_deltas = [3.0, -2.0, -1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625]
        #self.step_deltas = [1.0, -1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625]
        self.c1 = 0.005 #0.000000005 #0.000005 #0.000025
        self.c2 = 0.999 #0.75 #0.5#
        self.c3 = 0.75

    def do_forward_and_calc_params(self, testNet, xx, pp):
        meta = self.meta
        logits_k = testNet.forward_(xx)
        qq_k = softmax(np.transpose(logits_k.detach().cpu().numpy().copy()), axis=(0)) + self.epsilon
        qq_k_reduced = reduce_to_active(qq_k, pp)
        #loss_k = -np.sum(pp * np.log(qq_k))/meta.batch_size
        loss_k = -np.sum(np.log(qq_k_reduced))/meta.batch_size
        #diff_est_k = np.dot(ratio_k, delta_delta_est) #∇L(θ+α)
        return logits_k, qq_k, qq_k_reduced, loss_k
    
    def do_zero_step(self, testNet, xx, pp, momentum):
        self.paramProcessor.set_theta(testNet, momentum, 0.0, 0.0)
        #logging.info("##Bias_0 adjustment to {}:{}".format(eta_scale, testNet.conv1.bias[0].item()))
        return self.do_forward_and_calc_params(testNet, xx, pp)

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
            qq_one = softmax(np.transpose(logits_one.detach().cpu().numpy().copy()), axis=(0)) + self.epsilon
            delta_delta_one = reduce_to_active(qq - qq_one, pp) #We consider pp/qq without -
            diff_fact = np.dot(ratio, delta_delta_one)

            logits_k = None
            eta_scale = self.step_one
            ck0_armiho, ck0_wolf = 0.0, 1.0
            search_pos = 0
            #for step_delta in self.step_deltas:
            while True:
                if search_pos >= len(self.step_armiho) or search_pos < -len(self.step_wolf):
                    break
                if search_pos >= 0:
                    step_delta = self.step_armiho[search_pos]
                else:
                    step_delta = self.step_wolf[abs(1+search_pos)]
                if (step_delta != 0.0):
                    eta_scale += step_delta
                    self.paramProcessor.set_theta(testNet, momentum, eta, eta_scale)
                    #logging.info("##Bias_0 adjustment to {}:{}".format(eta_scale, testNet.conv1.bias[0].item()))
                logits_k, qq_k, qq_k_reduced, loss_k = self.do_forward_and_calc_params(testNet, xx, pp)
                ratio_k = reduce_to_active(pp/qq_k, pp)
                logging.info("##Step with eta_scale: {}, loss = {}".format(eta_scale, loss_k))
                logging.info("##Factual qq_k for ones: min={}, max={}, avg={}"\
                             .format(np.min(qq_k_reduced), np.max(qq_k_reduced), np.average(qq_k_reduced)))
                
                diff_fact_k = np.dot(ratio_k, delta_delta_one)
                pq_ratio_k = 1.0
                try:
                    pq_ratio_k = norm(1 - qq_k_reduced, ord=2)/norm(1 - qq_reduced, ord=2)
                except ValueError as ex:
                    logging.error("Error when (p-q)ratio norm-2 for ones calculated: {}".format(ex))
                logging.info("##With factual qq_k: diff_init = {}, diff_k = {}, (p-q)ratio norm-2 for ones: {}"\
                            .format(diff_fact, diff_fact_k, pq_ratio_k))
                #condition_armiho = loss_k - self.epsilon_criteria <= initialLoss + self.c1*eta_scale*diff_fact
                #condition_wolf = abs(diff_fact_k) - self.epsilon_criteria <= self.c2*abs(diff_fact)
                ck1_armiho = (loss_k - self.epsilon_criteria - initialLoss)/(eta_scale*diff_fact) #>=self.c1 when diff_fact < 0
                ck1_wolf = (abs(diff_fact_k) - self.epsilon_criteria)/abs(diff_fact) #<= self.c2
                condition_armiho = diff_fact < 0 and ck1_armiho >= self.c1
                condition_wolf = ck1_wolf <= self.c2
                condition_additional = diff_fact_k - self.epsilon_criteria <= self.c3*abs(diff_fact)
                logging.info("##Conditions: armiho = {}, wolf = {}, additional = {}, armiho_k={}, wolf_k={}"\
                             .format(condition_armiho, condition_wolf, condition_additional, ck1_armiho, ck1_wolf))
                if condition_armiho and condition_wolf and condition_additional:
                    break
                if condition_wolf and (not condition_armiho or not condition_additional) and search_pos>=0 and (ck1_armiho > ck0_armiho):
                    search_pos += 1
                    ck0_armiho, ck0_wolf = ck1_armiho, ck1_wolf
                    continue
                if condition_armiho and condition_additional and (not condition_wolf) and search_pos<=0 and (ck1_wolf < ck0_wolf):
                    search_pos -= 1
                    ck0_armiho, ck0_wolf = ck1_armiho, ck1_wolf
                    continue
                eta_scale = 0.0
                ck1_armiho, ck1_wolf = 0.0, 1.0
                logits_k, qq_k, qq_k_reduced, loss_k = self.do_zero_step(testNet, xx, pp, momentum)
                ratio_k = reduce_to_active(pp/qq_k, pp)
                diff_fact_k = np.dot(ratio_k, delta_delta_one)
                logging.info("##Step with eta_scale: {}, loss = {}".format(eta_scale, loss_k))
                logging.info("##With factual qq_k: diff_init = {}, diff_k = {}".format(diff_fact, diff_fact_k))                
                logging.info("##Conditions: armiho = {}, wolf = {}, additional = {}, armiho_k={}, wolf_k={}"\
                             .format(condition_armiho, condition_wolf, condition_additional, ck1_armiho, ck1_wolf))                
                break

        return eta_scale, logits_k, ck1_armiho, ck1_wolf

class StepResult:
    def __init__(self, eta_value, ck_armiho, ck_wolf, logits_k):
        self.eta_value = eta_value
        self.ck_armiho = ck_armiho
        self.ck_wolf = ck_wolf
        self.logits_k = logits_k

class OptimiserEtaSoftmaxArmihoBase:
    def __init__(self, meta, device, momentum=0.9, lbd_dict=None):
        self.meta = meta
        self.device = device
        self.momentum = momentum

        self.epsilon = 1e-8
        self.eta_min = self.epsilon
        self.eta_max = 10.0
        self.lbd_dict = lbd_dict
        #self.criterion = nn.CrossEntropyLoss()
        self.stepProcessor = StepProcessor(meta, self.epsilon)
        self.paramProcessor = self.stepProcessor.paramProcessor
        self.check_dropout = True

    def calc_eta(self, testNet, images, pp, qq):
        pass

    def step(self, testNet, labels, images, momentum, Nesterov = False, use_ones = False, use_fix = False):
        meta = self.meta
        pp = labels_to_softhot(labels, meta)
        #logging.info("##Bias_0 step-start:{}".format(testNet.conv1.bias[0].item()))

        self.paramProcessor.save_theta(testNet)
        if momentum > 0.0 and Nesterov == True and self.paramProcessor.is_delta_empty() == False:
            self.paramProcessor.set_theta(testNet, momentum, 0., 0.)
            #logging.info("##Bias_0 Nesterov pre-set:{}".format(testNet.conv1.bias[0].item()))

        testNet.zero_grad()
        logits = testNet.forward_(images)
        qq = (F.softmax(torch.transpose(logits, 0, 1), dim=0) + self.epsilon).to(meta.device)
        grad_buffer = calc_autograd(testNet, logits, meta)
        if not use_fix:
            eta_ones, eta_all = self.calc_eta(grad_buffer, pp, qq)
        
        logging.info("##Calculating params-delta")
        #testNet.zero_grad()
        #logits = testNet.forward_(xx)
        #loss = self.criterion(logits, labels)
        with torch.no_grad():
            grad_norm22 = self.paramProcessor.calc_ce_theta_delta(grad_buffer, pp, qq, self.lbd_dict)
        #logging.info("##Bias_0 calc_autograd:{}".format(self.paramProcessor.grad_current['conv1.bias'][0].item()))
        grad_lipsh_est = math.sqrt(grad_norm22) #/meta.batch_size
        loss = calc_ce_loss(pp, qq, meta)
        logging.info("##Min-grad value est = {}, grad_norm2^2={}, simple_step={}"\
                     .format(2*(1-self.stepProcessor.c1)/(grad_lipsh_est), grad_norm22, loss.item()/grad_norm22))

        eta, eta_scale = self.eta_min, 1.0
        ck_armiho, ck_wolf = 0.0, 1.0
        if not use_fix:
            eta = eta_ones if use_ones else eta_all
            #if meta.reduction == 'mean':
            #    eta = eta*meta.batch_size
                
            logging.info("##Eta value = {}".format(eta))
            if eta < self.eta_min or math.isnan(eta):
                logging.info("##Eta changed from {} to {}".format(eta, self.eta_min))
                eta = self.eta_min
            if eta > self.eta_max:
                logging.info("##Eta changed from {} to {}".format(eta, self.eta_max))
                eta = self.eta_max

            '''
            if self.check_dropout:
                do_dropout_val = testNet.do_dropout
                testNet.do_dropout = False
            eta_scale, logits_k, ck_armiho, ck_wolf = \
                self.stepProcessor.backtrack_armiho_wolf_additional(testNet, xx, pp, qq, eta, momentum)
            if self.check_dropout:
                testNet.do_dropout = do_dropout_val
            '''
        #else:
        with torch.no_grad():
            self.paramProcessor.set_theta(testNet, momentum, eta, 1.0)
            #logging.info("##Bias_0 initial 1.0:{}".format(testNet.conv1.bias[0].item()))
            logits_k = testNet.forward_(images)            

        self.paramProcessor.save_delta_current(momentum, eta, eta_scale)
        #logging.info("##Bias_0 delta:{}".format(self.paramProcessor.delta_current['conv1.bias'][0].item()))
        #logging.info("##Bias_0 step-finnish:{}".format(testNet.conv1.bias[0].item()))
        return StepResult(eta_scale*eta, ck_armiho, ck_wolf, logits_k)
    
class OptimiserEtaSoftmaxArmihoNorm2Base(OptimiserEtaSoftmaxArmihoBase):
    def __init__(self, meta, device, momentum=0.9, lbd_dict=None):
        super().__init__(meta, device, momentum, lbd_dict)

    def calc_eta(self, grad_buffer, pp, qq):
        with torch.no_grad():
            meta = self.meta
            delta = (pp - qq).to(meta.device)
            delta1 = reduce_to_active(delta, pp).to(meta.device)

            logging.info("##Calculating reduced softmax NTK")
            gamma = ntk_reduced(grad_buffer, delta, meta, self.lbd_dict)
            RR = ntk_softmax_reduced(gamma, qq, meta)
            RR1 = reduce_to_active(RR, pp)
            logging.info("##Calculating eta")
            eta = solve_eta_norm2(RR, delta) #for all n,α
            eta_ones = solve_eta_norm2(RR1, delta1) #for each α only n with pp=1 is taken
            logging.info("##Calculated eta for ones = {}, general eta = {}".format(eta_ones, eta))
            return eta_ones, eta

class OptimiserEtaSoftmaxArmihoNorm1Base(OptimiserEtaSoftmaxArmihoBase):
    def __init__(self, meta, device, momentum=0.9, lbd_dict=None):
        super().__init__(meta, device, momentum, lbd_dict)    

    def calc_eta(self, grad_buffer, pp, qq):
        with torch.no_grad():
            meta = self.meta
            delta = pp - qq
            delta1 = reduce_to_active(delta, pp)

            logging.info("##Calculating reduced softmax NTK")
            gamma = ntk_reduced(grad_buffer, delta, meta, self.lbd_dict)
            RR = ntk_softmax_reduced(gamma, qq, meta)
            RR1 = reduce_to_active(RR, pp)
            logging.info("##Calculating eta minimising norm_1 for delta all")
            fun = lambda eta: norm(torch.flatten(delta - eta*RR), ord=1)
            res = minimize_scalar(fun, bounds=(0, 1000))
            logging.info("##Optimal point for all found: {}".format(res))
            eta = res.x
            #for ones in softmax-encoding
            fun_ones = lambda eta_ones: norm(delta1 - eta_ones*RR1, ord=1)
            res_ones = minimize_scalar(fun_ones, bounds=(0, 1000))
            logging.info("##Optimal point for ones found: {}".format(res_ones))
            eta_ones = res_ones.x        
            logging.info("##Calculated eta for ones = {}, general eta = {}".format(eta_ones, eta))
            return eta_ones, eta
