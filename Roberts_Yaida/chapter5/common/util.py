import numpy as np
from scipy.special import softmax

import logging

class AverageMeter:
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(prediction, target):
    # Note that prediction.shape == target.shape == [B, ]
    matching = (prediction == target).float()
    return matching.mean().item()

def calculate_accuracy_np(prediction, target):
    # Note that prediction.shape == target.shape == [B, ]
    return np.average(prediction ==target)

def loss_crossentropy(zz_logits, true_labels):
    batch_size = len(true_labels)
    qq = softmax(zz_logits, axis=(0)) + 1e-8
    pp = np.zeros_like(qq)
    for col in range(batch_size):
        pp[true_labels[col],col]=1
    return -np.sum(pp * np.log(qq))/batch_size

def labels_to_onehot(zz_logits, true_labels):
    batch_size, output_dim = zz_logits.shape[0], zz_logits.shape[1]
    yy_onehot = np.zeros((output_dim, batch_size))
    for batch_num in range(batch_size):
        for output_num in range(output_dim):
            yy_onehot[output_num, batch_num] = \
                max(1.0, zz_logits[batch_num, output_num]) \
                    if output_num == true_labels[batch_num] \
                    else min(-1.0, zz_logits[batch_num, output_num])

    #logging.trace("For labels\n{}\nonehots are:\n{}".format(true_labels, yy_onehot))
    return yy_onehot
            
def labels_to_softhot(true_labels, output_dim):
    batch_size = true_labels.shape[0]
    yy_softhot = np.zeros((output_dim, batch_size))
    for batch_num in range(batch_size):
        yy_softhot[true_labels[batch_num], batch_num] = 1.0

    #logging.debug("For labels\n{}\nonehots are:\n{}".format(true_labels, yy_softhot))
    return yy_softhot
