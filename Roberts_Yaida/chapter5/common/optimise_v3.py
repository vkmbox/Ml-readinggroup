import math
import collections

import torch
from torch import nn
from torch.linalg import vector_norm
import torch.nn.functional as F

import numpy as np
from scipy.optimize import minimize_scalar

import logging

def reduce_to_active(matrix, pp):
    with torch.no_grad():
        return torch.sum(matrix*pp, 0)

def norm_fro(tensor):
    with torch.no_grad():
        return math.sqrt((torch.sum(tensor**2)).item())

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
    def __init__(self, logits, eta, eta_raw, ck_armiho=0.0, ck_wolf=0.0):
        self.logits = logits
        self.eta = eta
        self.eta_raw = eta_raw
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
        self.eta_cos_negative = 0.01
        self.eta0 = 0.000001

    """
    step_params: check_armiho=True/False (default True); check_additional=True/False (default True);
    estimation_type=iter-ce/iter-all-norm1/iter-all-norm2/iter-ones-norm1/iter-ones-norm2/analytic-norm2 (default iter-ce)
    """
    def step(self, labels, images, momentum, nesterov = False, step_params = None):
        net = self.net
        meta = self.meta
        if net.training:
            raise ValueError("net.training must be False")
        pp = labels_to_softhot(labels, meta)
        #logging.info("##Bias_0 step-start:{}".format(testNet.conv1.bias[0].item()))
        self.paramProcessor.save_theta(net)

        diff_initial, momentum_coeff = 0.0, 1.0
        for ii in range(2):
            momentum = momentum*momentum_coeff
            if momentum > 0.0 and nesterov == True and self.paramProcessor.is_delta_empty() == False:
                self.paramProcessor.set_theta(net, momentum, 0., 0.)

            logging.info("##Calculating params-delta")
            net.zero_grad()
            logits = self.do_forward(images, 'toss') ##TODO: new dropout is generated here (1*)
            loss = self.criterion(logits, labels)
            grad_norm2 = math.sqrt(self.paramProcessor.calc_autograd(net, loss, self.lbd_dict))
            logging.info("##Gradient norm2:{}".format(grad_norm2))
            if self.training_mode:
                logits = self.do_forward(images, 'train') ## all qqxx calculated with dropout off

            with torch.no_grad():
                #Eta-calculation
                qq0 = self.softmax(logits, meta) #q(t)
                loss_initial = crossentropy_avg(pp, qq0)
                logging.info("##Loss initial:{}".format(loss_initial))
                eta_test = self.eta0 #small step-size
                self.paramProcessor.set_theta(net, momentum, eta_test) #small step
                logits_test = self.do_forward(images, 'train') ##TODO: no new dropout generated here, a generated in (1*) must be used
                qq_test = self.softmax(logits_test, meta) #q(t+1)
                pq_cos = self.pq_cos(pp, qq0, qq_test)
                if pq_cos >= 0.0:
                    eta_coeff = self.eta_coeff(eta_test, pp, qq0, qq_test, step_params)
                    eta_raw = eta_test*eta_coeff*momentum_coeff
                    logging.info("##Eta raw-value = {} with momentum_coeff = {}".format(eta_raw, momentum_coeff))
                    eta = eta_raw = self.eta_bounded(eta_raw)
                else:
                    eta = eta_raw = self.eta_cos_negative
                
                #Step with scale 1.0
                self.paramProcessor.set_theta(net, momentum, eta, 1.0)
                logits1 = self.do_forward(images, 'train') ##TODO: no new dropout generated here, a generated in (1*) must be used
                qq1 = self.softmax(logits1, meta)
                #Linearity estimation
                iter_num, iter_cond=0, pq_cos > 0.0
                while iter_cond: #TODO: cos may go to < 0 during iterations!
                    coeff_correction = self.eta_coeff_analytic_n2(1.0, pp, qq0, qq1)
                    logging.info("##Eta raw-value {} corrected to {} with coeff {}".format(eta_raw, eta_raw*coeff_correction, coeff_correction))
                    eta_raw = eta_raw*coeff_correction
                    #eta = eta*cos_phi2
                    self.paramProcessor.set_theta(net, momentum, eta_raw, 1.0)
                    logits1 = self.do_forward(images, 'train') ##TODO: no new dropout generated here, a generated in (1*) must be used
                    qq1 = self.softmax(logits1, meta)
                    iter_num += 1
                    if (iter_num >= self.iter_max) or (2/3 < coeff_correction and coeff_correction < 3/2):
                        logging.info("##Finall correction-coeff value: {}".format(coeff_correction))
                        eta = self.eta_bounded(eta_raw)
                        iter_cond = False
                        if eta != eta_raw:
                            self.paramProcessor.set_theta(net, momentum, eta, 1.0)
                            logits1 = self.do_forward(images, 'train') ##TODO: no new dropout generated here, a generated in (1*) must be used
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
                    #momentum = momentum*momentum_coeff

        if self.qu is None:
            self.qu = collections.deque(5*[diff_initial], 5)
        else:
            self.qu.append(diff_initial)

        eta_scale, logits, ck1_armiho, ck1_wolf = \
            self.step_reduction(images, pp, qq0, qq1, logits1, loss_initial, diff_initial, momentum, eta, step_params) \
                if pq_cos > 0.0 and (self.get_param(step_params, 'check_armiho', True) == True or self.get_param(step_params, 'check_additional', True) == True) \
                    else self.step_one(logits1)

        logging.info("##Eta-value after conditions are applied: {}".format(eta*eta_scale))
        self.paramProcessor.save_delta_current(momentum, eta, eta_scale)
        return StepResult(logits, eta*eta_scale, eta_raw, ck1_armiho, ck1_wolf)
    
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
                    logits_k = self.do_forward(images, 'train') ##TODO: no new dropout generated here, a generated in (1*) must be used
                qq = self.softmax(logits_k, meta)
                loss_k = crossentropy_avg(pp, qq)
                #diff_k = (torch.sum((pp/qq)*(qq0-qq))).item()
                diff_k = (torch.sum((pp/qq)*(qq0-qq1))).item()/pp.shape[1]
                #condition_armiho = loss_k - self.epsilon_criteria <= initialLoss + self.c1*eta_scale*diff_initial
                #condition_wolf = abs(diff_k) - self.epsilon_criteria <= self.c2*abs(diff_initial)
                ck1_armiho = (loss_k - self.epsilon_criteria - loss_initial)/(eta_scale*diff_initial) #>=self.c1 when diff_initial < 0
                ck1_wolf = (abs(diff_k) - self.epsilon_criteria)/abs(diff_initial) #<= self.c2
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

    def pq_cos(self, pp, qq0, qq_test):
        with torch.no_grad():
            delta_pq, delta_qq = pp-qq0, qq_test-qq0
            norm_pq, norm_qq = norm_fro(delta_pq), norm_fro(delta_qq)
            cos_phi = (torch.sum(delta_pq*delta_qq)/(norm_pq*norm_qq)).item()
            logging.info("##cos(pp^qq):{}".format(cos_phi))
            return cos_phi

    def eta_coeff_analytic_n2(self, eta0, pp, qq0, qq_test):
        with torch.no_grad():
            logging.info("##--==Analytic norm_2 formula params==--")
            delta_pq, delta_qq = pp-qq0, qq_test-qq0
            norm_pq, norm_qq = norm_fro(delta_pq), norm_fro(delta_qq)
            cos_phi1 = self.pq_cos(pp, qq0, qq_test)
            return math.sqrt(max(((norm_pq*cos_phi1)/(norm_qq*eta0 + self.epsilon)), 0.0))
        
    def eta_coeff_iter(self, eta0, pp, qq0, qq_test, norm_ord, ones):
        with torch.no_grad():
            summand1, summand2 = qq0 - pp, qq_test - qq0
            fun = None
            if ones:
                fun = lambda coeff: (vector_norm(reduce_to_active(summand1 + coeff*summand2, pp), ord=norm_ord)).item()
            else:
                fun = lambda coeff: (vector_norm(summand1 + coeff*summand2, ord=norm_ord)).item()
            res = minimize_scalar(fun, bounds=(0, 100/eta0))
            return res.x

    def eta_coeff_crossentropy(self, eta0, pp, qq0, qq_test):
        M_const = 1e3
        with torch.no_grad():
            pp_a = reduce_to_active(pp, pp)
            qq0_a = reduce_to_active(qq0, pp)
            qqtest_a = reduce_to_active(qq_test, pp)
            logging.info("##\n --==QQ initial: min={}, max={}, avg={}==--"\
                            .format(torch.min(qq0_a), torch.max(qq0_a), torch.mean(qq0_a)))            
            zeros = torch.zeros_like(pp_a)
            epsilons = torch.full_like(pp_a, self.epsilon)
            summand = qqtest_a - qq0_a
            fun = lambda coeff: (torch.sum(-pp_a *torch.log(torch.max(qq0_a + coeff*summand, epsilons)))\
                                 +M_const*torch.sum(torch.max(qq0_a + coeff*summand, zeros))).item()
            res = minimize_scalar(fun, bounds=(0, min(1.0, self.eta_max)/eta0))
            qq_active = qq0_a + res.x*summand
            logging.info("##\n --==QQ estimated: min={}, max={}, avg={}==--"\
                            .format(torch.min(qq_active), torch.max(qq_active), torch.mean(qq_active)))
            return res.x
    
    def softmax(self, logits, meta):
        with torch.no_grad():
            return (F.softmax(torch.transpose(logits, 0, 1), dim=0) + self.epsilon).to(meta.device)

    def eta_bounded(self, eta):
        if eta > self.eta_max:
            logging.info("##Eta-value is reduced from {} to {}".format(eta, self.eta_max))
            eta = self.eta_max
        if eta < self.eta_min:
            logging.info("##Eta-value is increased from {} to {}".format(eta, self.eta_min))
            eta = self.eta_min
        return eta
    
    def eta_coeff(self, eta_test, pp, qq0, qq_test, step_params):
        coeff_ayc2 = self.eta_coeff_analytic_n2(eta_test, pp, qq0, qq_test)
        coeff_all_itr1 = self.eta_coeff_iter(eta_test, pp, qq0, qq_test, 1, ones=False)
        coeff_all_itr2 = self.eta_coeff_iter(eta_test, pp, qq0, qq_test, 2, ones=False)
        coeff_ones_itr1 = self.eta_coeff_iter(eta_test, pp, qq0, qq_test, 1, ones=True)
        coeff_ones_itr2 = self.eta_coeff_iter(eta_test, pp, qq0, qq_test, 2, ones=True)
        coeff_crossentropy = self.eta_coeff_crossentropy(eta_test, pp, qq0, qq_test)
        logging.info("##--==Eta-value estimations==--\n"+\
                        " analytic={}, all-norm1={}, all-norm2={}, ones-norm1={}, ones-norm2={}, crossentropy={}"\
                        .format(eta_test*coeff_ayc2, eta_test*coeff_all_itr1, eta_test*coeff_all_itr2\
                                , eta_test*coeff_ones_itr1, eta_test*coeff_ones_itr2, eta_test*coeff_crossentropy))
        eta_type = self.get_param(step_params, 'estimation_type', 'iter-norm2')
        #iter-all-norm1/iter-all-norm2/iter-ones-norm1/iter-ones-norm2
        if eta_type == 'analytic-norm2':
            return coeff_ayc2
        elif eta_type == 'iter-ones-norm1':
            return coeff_ones_itr1
        elif eta_type == 'iter-ones-norm2':
            return coeff_ones_itr2
        elif eta_type == 'iter-all-norm1':
            return coeff_all_itr1
        elif eta_type == 'iter-all-norm2':
            return coeff_all_itr2
        elif eta_type == 'iter-ce':
            return coeff_crossentropy
        else:
            return coeff_all_itr2

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
