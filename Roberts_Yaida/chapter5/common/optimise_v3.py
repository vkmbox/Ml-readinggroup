import math
import torch
from torch import nn
from torch.linalg import matrix_norm
import torch.nn.functional as F

import logging

def norm_fro(tensor):
    return math.sqrt((torch.sum(tensor**2)).item())

def crossentropy_avg(pp, qq):
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
        #grad_norm2_squared = 0.
        ii = 0
        for name in param_buffer:
            lambda_value = lambda_dict.get(name, 1.) if lambda_dict is not None else 1.
            grad = df[ii].detach().clone()
            self.grad_current[name] = lambda_value * grad
            #grad_norm2_squared += ((grad)**2).sum().item()
            ii += 1

class StepResult:
    def __init__(self, logits, eta, ck_armiho=0.0, ck_wolf=0.0):
        self.logits = logits
        self.eta = eta
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
        self.iterations_max = 10

    #Armiho: Loss(θ+α) <= c1*α*∇Loss(θ) + Loss(θ)
    #Wolf: |∇Loss(θ+α)| <= c2*|∇Loss(θ)|
    #0<c1<c2<1
    #Additional condition: ∇Loss(θ+α) <= c3*|∇Loss(θ)| , 0<c3<1 (Significant loss growth at the final point is unacceptable)
        self.c1 = 0.0001
        self.c2 = 0.999
        self.c3 = 0.75
        self.armiho_beta = 0.5

        self.eta_min = 0.0001
        self.eta_max = 1.0
        self.eta0 = 0.000001

    def step(self, labels, images, momentum, nesterov = False):
        net = self.net
        meta = self.meta
        pp = labels_to_softhot(labels, meta)
        #logging.info("##Bias_0 step-start:{}".format(testNet.conv1.bias[0].item()))

        self.paramProcessor.save_theta(net)
        if momentum > 0.0 and nesterov == True and self.paramProcessor.is_delta_empty() == False:
            self.paramProcessor.set_theta(net, momentum, 0., 0.)

        logging.info("##Calculating params-delta")
        net.zero_grad()
        logits = net.forward_(images) ##TODO: new dropout is generated here (1*)
        loss = self.criterion(logits, labels)
        self.paramProcessor.calc_autograd(net, loss, self.lbd_dict)
        with torch.no_grad():
            qq0 = (F.softmax(torch.transpose(logits, 0, 1), dim=0) + self.epsilon).to(meta.device) #q(t)
            loss_initial = crossentropy_avg(pp, qq0)
            logging.info("##Loss initial:{}".format(loss_initial))
            eta0 = self.eta0 #small step-size
            self.paramProcessor.set_theta(net, momentum, eta0) #small step
            logits = net.forward_(images) ##TODO: no new dropout generated here, a generated in (1*) must be used
            qq1 = (F.softmax(torch.transpose(logits, 0, 1), dim=0) + self.epsilon).to(meta.device) #q(t+1)
            delta_pq, delta_qq = pp-qq0, qq1-qq0
            norm_pq, norm_qq = norm_fro(delta_pq), norm_fro(delta_qq)
            cos_phi = torch.sum(delta_pq*delta_qq)/(norm_pq*norm_qq)
            logging.info("##norm(PQ)={},norm(QQ)={}".format(norm_pq, norm_qq))
            logging.info("##PQ*QQ:{}".format(torch.sum(delta_pq*delta_qq)))
            logging.info("##cos(fi):{}".format(cos_phi))
            coeff = math.sqrt(max(((norm_pq*cos_phi)/(norm_qq*eta0)), 0.0))
            eta = eta0*coeff
            logging.info("##Eta-value is found: {}".format(eta))
            if eta > self.eta_max:
                logging.info("##Eta-value is reduced from {} to {}".format(eta, self.eta_max))
                eta = self.eta_max
            if eta < self.eta_min:
                logging.info("##Eta-value is increased from {} to {}".format(eta, self.eta_min))
                eta = self.eta_min

            #self.paramProcessor.set_theta(net, momentum, eta, 1.0)
            eta_scale, logits, ck1_armiho, ck1_wolf = self.step_reduction(images, pp, qq0, loss_initial, momentum, eta)
            self.paramProcessor.save_delta_current(momentum, eta, eta_scale)
            return StepResult(logits, eta*eta_scale, ck1_armiho, ck1_wolf)
        
    def step_reduction(self, images, pp, qq0, loss_initial, momentum, eta):
        net = self.net
        meta = self.meta
        logits_k = None
        eta_scale, diff_initial = 1.0, 0.0
        with torch.no_grad():
            while eta_scale > 0.01:
                self.paramProcessor.set_theta(net, momentum, eta, eta_scale)
                logits_k = net.forward_(images) ##TODO: no new dropout generated here, a generated in (1*) must be used
                qq = (F.softmax(torch.transpose(logits_k, 0, 1), dim=0) + self.epsilon).to(meta.device)
                loss_k = crossentropy_avg(pp, qq)
                diff_k = torch.sum((pp/qq)*(qq0-qq))
                if eta_scale == 1.0:
                    diff_initial = torch.sum((pp/qq0)*(qq0-qq))
                #condition_armiho = loss_k - self.epsilon_criteria <= initialLoss + self.c1*eta_scale*diff_initial
                #condition_wolf = abs(diff_k) - self.epsilon_criteria <= self.c2*abs(diff_initial)
                ck1_armiho = (loss_k - self.epsilon_criteria - loss_initial)/(eta_scale*diff_initial) #>=self.c1 when diff_initial < 0
                ck1_wolf = (abs(diff_k) - self.epsilon_criteria)/abs(diff_initial) #<= self.c2
                condition_armiho = diff_initial < 0 and ck1_armiho >= self.c1
                condition_wolf = ck1_wolf <= self.c2
                condition_additional = diff_k - self.epsilon_criteria <= self.c3*abs(diff_initial)
                logging.info("##Step with eta_scale: {}, loss = {}, diff_init = {}, diff_k = {}"\
                             .format(eta_scale, loss_k, diff_initial, diff_k))
                logging.info("##Conditions: armiho = {}, wolf = {}, additional = {}, armiho_k={}, wolf_k={}"\
                             .format(condition_armiho, condition_wolf, condition_additional, ck1_armiho, ck1_wolf))
                if condition_armiho:
                    break
                eta_scale = eta_scale*self.armiho_beta
            
            logging.info("##Eta-value after conditions are applied: {}".format(eta*eta_scale))
            return eta_scale, logits_k, ck1_armiho, ck1_wolf
