import math
import collections

import torch
from torch import nn
#from torch.linalg import vector_norm
import torch.nn.functional as F

import numpy as np

import logging

def sign(number):
    return (-1.0 if number < 0.0 else 1.0)

def reduce_to_active(matrix, pp):
    with torch.no_grad():
        return torch.sum(matrix*pp, 0)
    
def norm_fro(tensor, ord=2):
    with torch.no_grad():
        return math.pow((torch.sum(torch.abs(tensor)**ord)).item(), 1/ord)

def crossentropy_avg(pp, qq):
    with torch.no_grad():
        batch_size = pp.shape[1]
        loss = -torch.sum(pp * torch.log(qq))/batch_size
        return loss.item()

def labels_to_softhot(true_labels, meta):
    batch_size = true_labels.shape[0]
    with torch.no_grad():
        pp = torch.zeros(meta.output_dim, batch_size).to(meta.device)
        for batch_num in range(batch_size):
            pp[true_labels[batch_num], batch_num] = 1.0
    return pp

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
                    raise ValueError("eta_scale or momentum must be in interval(0., 1.). momentum={}, eta={}, eta_scale={}"\
                                     .format(momentum, eta, eta_scale))

    def save_delta_current(self, momentum, eta, eta_scale = 1.0):
        with torch.no_grad():
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
        logging.info("##Autograd start")
        df = torch.autograd.grad(loss, param_buffer.values())#, retain_graph=True, create_graph=True, allow_unused=True)
        logging.info("##Autograd finish")
        with torch.no_grad():
            norm2_squared = 0.
            ii = 0
            for name in param_buffer:
                lambda_value = lambda_dict.get(name, 1.) if lambda_dict is not None else 1.
                grad = df[ii].detach().clone()
                self.grad_current[name] = lambda_value * grad
                norm2_squared += ((grad)**2).sum().item()
                ii += 1

            return norm2_squared

class StepResult:
    def __init__(self, logits, eta, eta_raw, eta_ratio, ck_armiho=0.0, ck_wolf=0.0):
        self.logits = logits
        self.eta = eta
        self.eta_raw = eta_raw
        self.eta_ratio = eta_ratio
        self.ck_armiho = ck_armiho
        self.ck_wolf = ck_wolf

class NetLineStepProcessor:
    def __init__(self, net, criterion, meta, device, lbd_dict=None):
        self.net = net
        self.meta = meta
        self.device = device
        self.epsilon = 1e-9
        self.epsilon_criteria = 1e-5
        self.lbd_dict = lbd_dict
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.paramProcessor = ParameterProcessor()
        self.momentum_gradient_smoothing_coefficient = 0.0
        self.qu = None
        self.training_mode = False
        self.iter_max = 2

    #Armiho: Loss(θ+α) <= c1*α*∇Loss(θ) + Loss(θ)
    #Wolf: |∇Loss(θ+α)| <= c2*|∇Loss(θ)|
    #0<c1<c2<1
    #Additional condition: ∇Loss(θ+α) <= c3*|∇Loss(θ)| , 0<c3<1 (Significant loss growth at the final point is unacceptable)
        self.c1 = 0.0001
        self.c2 = 0.999
        self.c3 = 0.25
        self.armiho_beta = 0.5

        self.eta_min = 0.0001
        self.eta_max = 1.0
        self.alpha = 0.25
        self.beta = 0.01
        self.eta0 = 0.0001

    """
    step_params: check_armiho=True/False (default True); check_additional=True/False (default True);
    """
    def step(self, labels, images, momentum, nesterov = False, step_params = None):
        net = self.net
        meta = self.meta
        if net.training:
            raise ValueError("net.training must be False")
        pp = labels_to_softhot(labels, meta)
        self.paramProcessor.save_theta(net)

        diff_initial, momentum_coeff, eta_ratio = 0.0, 1.0, 0.0
        for ii in range(2):
            momentum = momentum*momentum_coeff
            if momentum > 0.0 and nesterov == True and self.paramProcessor.is_delta_empty() == False:
                self.paramProcessor.set_theta(net, momentum, 0., 0.)

            logging.info("##Calculating params-delta")
            net.zero_grad()
            logits = self.do_forward(images, 'toss') ## new dropout is generated here (1*)
            loss = self.criterion(logits, labels)
            grad_norm2 = math.sqrt(self.paramProcessor.calc_autograd(net, loss, self.lbd_dict))
            logging.info("##Gradient norm2:{}".format(grad_norm2))
            with torch.no_grad():
                if self.training_mode:
                    logits = self.do_forward(images, 'train') ## all qqxx calculated with dropout off
                #Eta-calculation
                qq0 = self.softmax(logits, meta) #q(t)
                loss_initial = crossentropy_avg(pp, qq0)
                logging.info("##Loss initial:{}".format(loss_initial))
                eta_curr, eta_next = self.eta0, 0. #small step-size η0, η1, k ← ηtest, 0, 0
                iter_num, iter_cond =  0, True
                while iter_cond:
                    self.paramProcessor.set_theta(net, momentum, eta_curr, 1.0) #small step
                    logits1 = self.do_forward(images, 'train') ## no new dropout generated here, a generated in (1*) must be used
                    qq1 = self.softmax(logits1, meta) #q(t+1)
                    eta_next = self.eta_analytic_n2(eta_curr, pp, qq0, qq1)
                    eta_ratio = eta_next/(eta_curr + self.epsilon)
                    logging.info("##--==On iter {} for eta_curr={} eta_next={} with ratio={} ==--"\
                                 .format(iter_num, eta_curr, eta_next, eta_ratio))
                    iter_num += 1
                    if (iter_num >= self.iter_max) or (0.5 < abs(eta_ratio) and abs(eta_ratio) < 2.0):
                        logging.info("##Finall selected on iter={} eta_raw={} with eta_next={} next/raw ratio={} and momentum_coeff={}"\
                                     .format(iter_num, eta_curr, eta_next, eta_ratio, momentum_coeff))
                        iter_cond = False
                    else:
                        eta_curr = eta_next

                eta = self.eta_bounded(eta_curr*momentum_coeff)
                if eta != eta_curr:
                    self.paramProcessor.set_theta(net, momentum, eta, 1.0)
                    logits1 = self.do_forward(images, 'train') ## no new dropout generated here, a generated in (1*) must be used
                    qq1 = self.softmax(logits1, meta)

                diff_initial = (torch.sum((pp/qq0)*(qq0-qq1))).item()/pp.shape[1]
                #Momentum gradient-smoothing
                if ii == 0:
                    diff_average = 1. if self.qu is None else np.average(self.qu)
                    if momentum <= 0. or diff_initial >= 0. or diff_average >=0. \
                        or self.momentum_gradient_smoothing_coefficient == 0. or self.qu is None \
                            or diff_average*self.momentum_gradient_smoothing_coefficient <= diff_initial: #Differentials expected to be <0!
                        break
                    logging.info("##Average over 5 previous diff_init: {}, current diff_init: {}".format(diff_average, diff_initial))
                    momentum_coeff = math.sqrt(diff_average*self.momentum_gradient_smoothing_coefficient/diff_initial)
                    logging.info("##Momentum coeff = {}, momentum is reduced from {} to {}"\
                                    .format(momentum_coeff, momentum, momentum*momentum_coeff))

        if self.qu is None:
            self.qu = collections.deque(5*[diff_initial], 5)
        else:
            self.qu.append(diff_initial)

        eta_scale, logits, ck1_armiho, ck1_wolf = \
            self.step_reduction(images, pp, qq0, qq1, logits1, loss_initial, diff_initial, momentum, eta, step_params) \
                if (self.get_param(step_params, 'check_armiho', True) == True or self.get_param(step_params, 'check_additional', True) == True) \
                    else self.step_one(logits1)

        logging.info("##Eta-value after conditions are applied: {}".format(eta*eta_scale))
        self.paramProcessor.save_delta_current(momentum, eta, eta_scale)
        return StepResult(logits, eta*eta_scale, eta_curr*momentum_coeff, eta_ratio, ck1_armiho, ck1_wolf)
    
    def get_param(self, step_params, param_name, default):
        if step_params is None:
            return default
        return step_params.get(param_name, default)

    def step_one(self, logits1):
        return 1.0, logits1, 1.0, 0.0

    def step_reduction(self, images, pp, qq0, qq1, logits1, loss_initial, diff_initial, momentum, eta, step_params):
        net = self.net
        meta = self.meta
        logits_k = None
        eta_scale = 1.0
        with torch.no_grad():
            while eta_scale > 0.001:
                if eta_scale == 1.0:
                    logits_k = logits1
                else:
                    self.paramProcessor.set_theta(net, momentum, eta, eta_scale)
                    logits_k = self.do_forward(images, 'train') ## no new dropout generated here, a generated in (1*) must be used
                qq = self.softmax(logits_k, meta)
                loss_k = crossentropy_avg(pp, qq)
                diff_k = (torch.sum((pp/qq)*(qq0-qq1))).item()/pp.shape[1]
                #condition_armiho = loss_k - self.epsilon_criteria <= initialLoss + self.c1*eta_scale*diff_initial
                #condition_wolf = abs(diff_k) - self.epsilon_criteria <= self.c2*abs(diff_initial)
                ck1_armiho = (loss_k - self.epsilon_criteria - loss_initial)/(eta_scale*diff_initial+self.epsilon) #>=self.c1 when diff_initial < 0
                ck1_wolf = (abs(diff_k) - self.epsilon_criteria)/abs(diff_initial+self.epsilon) #<= self.c2
                condition_armiho = diff_initial < 0 and ck1_armiho >= self.c1
                condition_wolf = ck1_wolf <= self.c2
                condition_additional = diff_k - self.epsilon_criteria <= self.c3*abs(diff_initial)
                logging.info("##Step with eta_scale: {}, loss = {}, df(0) = {}, df({}) = {}"\
                             .format(eta_scale, loss_k, diff_initial, eta_scale, diff_k))
                logging.info("##Conditions: armiho = {}, wolf = {}, additional = {}, armiho_k={}, wolf_k={}"\
                             .format(condition_armiho, condition_wolf, condition_additional, ck1_armiho, ck1_wolf))
                if (self.get_param(step_params, 'check_armiho', True) == False or condition_armiho) \
                    and (self.get_param(step_params, 'check_additional', True) == False or condition_additional):
                    break
                eta_scale = eta_scale*self.armiho_beta
            
            return eta_scale, logits_k, ck1_armiho, ck1_wolf

    def pq_cos(self, vector1, vector2):
        with torch.no_grad():
            norm1, norm2 = norm_fro(vector1, ord=2), norm_fro(vector2, ord=2)
            cos_phi = (torch.sum(vector1*vector2)/(norm1*norm2 + self.epsilon)).item()
            return cos_phi

    def eta_analytic_n2(self, eta_test, pp, qq0, qq_test):
        with torch.no_grad():
            delta_pq, delta_qq = pp-qq0, qq_test-qq0
            norm_pq, norm_qq = norm_fro(delta_pq, ord=2), norm_fro(delta_qq, ord=2)
            cos_phi1 = self.pq_cos(delta_pq, delta_qq)
            logging.info("##cos(pp^qq)={}, norm_pq={}, norm_qq={}".format(cos_phi1, norm_pq, norm_qq))
            eta_next = sign(cos_phi1)*math.sqrt(abs(((norm_pq*cos_phi1*eta_test*self.alpha)/(norm_qq + self.beta))))
            logging.info("##Eta-value estimations: analytic={}".format(eta_next))
            return eta_next

    def softmax(self, logits, meta):
        with torch.no_grad():
            return (F.softmax(torch.transpose(logits, 0, 1), dim=0) + self.epsilon).to(meta.device)

    def eta_bounded(self, eta):
        if abs(eta) > self.eta_max:
            logging.info("##Eta-value is reduced from {} to {}".format(eta, self.eta_max*sign(eta)))
            eta = self.eta_max*sign(eta)
        if abs(eta) < self.eta_min:
            logging.info("##Eta-value is increased from {} to {}".format(eta, self.eta_min*sign(eta)))
            eta = self.eta_min*sign(eta)
        return eta
    
    #dropout_mode = 'eval' #toss/train/eval
    def do_forward(self, images, dropout_mode):
        net = self.net
        if self.training_mode and dropout_mode == 'toss':
            training = net.training
            net.train(True)
            logging.info("##--==Train forward==--")
            logits = net.forward(images)
            net.train(training)
            return logits
        else:
            return net.forward(images)
