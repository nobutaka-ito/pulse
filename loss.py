import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import convolve

def sigmoid_loss(z):

    return F.sigmoid(-z)


def sa_loss(yhat, mix_stft, cln_stft):

    return torch.mean((F.sigmoid(yhat) * torch.abs(mix_stft) - torch.abs(cln_stft)) ** 2)


def weighted_pu_loss(y, yhat, mix_stft, beta, gamma = 1.0, prior = 0.5, ell = sigmoid_loss, p = 1, mode = 'nn'):

    epsilon = 10 ** -7
    weight = (torch.abs(mix_stft) + epsilon) ** p
    pos = prior * torch.sum(y * ell(yhat) * weight) / (torch.sum(y) + epsilon)
    neg = torch.sum((1. - y) * ell(-yhat) * weight) / (torch.sum(1. - y) + epsilon) - prior * torch.sum(y * ell(-yhat) * weight) / (torch.sum(y) + epsilon)

    if mode == 'unbiased':
        loss = pos + neg
    elif mode == 'nn':
        if neg > beta:
            loss = pos + neg
        else:
            loss = gamma * (beta - neg)

    return loss


def MixIT_loss(yhat, mix_stft, noise_stft, noisy_stft):

    loss1 = torch.mean(((F.sigmoid(yhat[:,0,:,:]) + F.sigmoid(yhat[:,1,:,:])) * torch.abs(mix_stft) - torch.abs(noisy_stft)) ** 2)
    loss1 += torch.mean((F.sigmoid(yhat[:,2,:,:])  * torch.abs(mix_stft) - torch.abs(noise_stft)) ** 2)
    loss2 = torch.mean(((F.sigmoid(yhat[:,0,:,:]) + F.sigmoid(yhat[:,2,:,:])) * torch.abs(mix_stft) - torch.abs(noisy_stft)) ** 2)
    loss2 += torch.mean((F.sigmoid(yhat[:,1,:,:])  * torch.abs(mix_stft) - torch.abs(noise_stft)) ** 2)
    
    return torch.minimum(loss1, loss2)