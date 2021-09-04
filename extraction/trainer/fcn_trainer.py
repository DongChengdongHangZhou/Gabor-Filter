import datetime
import itertools
import os
import os.path as osp
import shutil

import numpy as np
import pytz
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import pdb

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


class FCNTrainer(object):
    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter,
                 size_average=False):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer
	self.MSELoss = nn.MSELoss()
	self.NLLLoss2d = nn.NLLLoss2d()

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
        self.size_average = size_average

	self.best_ori_loss = 1

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/ori_loss',
            'train/period_loss',
            'train/mask_loss',
	    'train/seg_loss'
            'valid/ori_loss',
            'valid/period_loss',
            'valid/mask_loss',
	    'valid/seg_loss',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter

    def validate(self):
        self.model.eval()
        self.iteration = self.epoch * len(self.train_loader)

        val_loss = 0
        metrics = []
        visualizations = []
	sum_oloss, sum_ploss, sum_mloss, sum_sacc = 0,0,0,0
        for batch_idx, (img, ori, period, mask) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid epoch=%d' % self.epoch, ncols=70, leave=False):
            if self.cuda:
                img, ori, period, mask = img.cuda(), ori.cuda(), period.cuda(), mask.cuda()
            img, ori, period, mask = Variable(img, volatile=True),Variable(ori),Variable(period),Variable(mask)
            score = self.model(img)
            loss1 = self.ori_loss(score[0], ori, mask)
	    loss2 = self.period_loss(score[1], period, mask)
	    loss3 = self.mask_loss(score[2], mask)
	    loss = loss1 + loss2 + loss3

            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')

	    SAcc = self.seg_acc(score[2], mask)
	    OLoss = float(loss1.data[0])
	    PLoss = float(loss2.data[0])
            MLoss = float(loss3.data[0])
	   
  	    sum_oloss += OLoss
	    sum_ploss += PLoss
	    sum_mloss += MLoss
	    sum_sacc += SAcc

	    print 'OLoss:{0:.4f}, PLoss:{1:.4f}, MLoss:{2:.4f}, SAcc:{3:.3f}'.format(OLoss, PLoss, MLoss, SAcc)

  	sum_oloss /= len(self.val_loader)
	sum_ploss /= len(self.val_loader)
	sum_mloss /= len(self.val_loader)
	sum_sacc /= len(self.val_loader)
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai')) - self.timestamp_start
            log = [self.epoch, self.iteration] + [''] * 4 + [sum_oloss,sum_ploss,sum_mloss,sum_sacc] + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

	is_best = self.best_ori_loss > sum_oloss
        if is_best:
            self.best_ori_loss = sum_oloss
        torch.save({
            'epoch': self.epoch,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
	    'best_ori_loss': self.best_ori_loss
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

    def train_epoch(self):
        self.model.train()
	
        for batch_idx, (img, ori, period, mask) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch={0}'.format(self.epoch), ncols=70, leave=False):

            self.iteration = batch_idx + self.epoch * len(self.train_loader)

            if self.cuda:
                img, ori, period, mask = img.cuda(), ori.cuda(), period.cuda(), mask.cuda()
            img, ori, period, mask = Variable(img), Variable(ori), Variable(period), Variable(mask)

            self.optim.zero_grad()
            score = self.model(img)
            loss1 = self.ori_loss(score[0], ori, mask)
	    loss2 = self.period_loss(score[1], period, mask)
	    loss3 = self.mask_loss(score[2], mask)
	    loss = loss1 + loss2 + loss3
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()
	    
	    SAcc = self.seg_acc(score[2], mask)
	    OLoss = float(loss1.data[0])
	    PLoss = float(loss2.data[0])
            MLoss = float(loss3.data[0])

	    print 'OLoss:{0:.4f}, PLoss:{1:.4f}, MLoss:{2:.4f}, SAcc:{3:.3f}'.format(OLoss, PLoss, MLoss, SAcc)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai')) - self.timestamp_start
                log = [self.epoch, self.iteration] + [loss1.data[0], loss2.data[0], loss3.data[0], SAcc] + [''] * 4 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def train(self):
        for epoch in itertools.count(self.epoch):
            self.epoch = epoch
            self.validate()
            if self.iteration >= self.max_iter:
                break
            self.train_epoch()

    def mask_loss(self, pred, target):
        log_p = F.log_softmax(pred)
        loss = self.NLLLoss2d(log_p, target)
        return loss

    def ori_loss(self, pred, target, mask):
	# normalize the output
        cos2th = pred[:, 0, :, :]
        sin2th = pred[:, 1, :, :]
        norm = torch.sqrt(cos2th * cos2th + sin2th * sin2th)
        cos2th = cos2th / norm
        sin2th = sin2th / norm
	# compute the cos(theta1-theta2)
	cos_2dif = cos2th * target[:, 0, :, :] + sin2th * target[:, 1, :, :]
	# take the foreground
	foreground = cos_2dif[mask==1]
	loss = 1 - torch.sum(foreground) / len(foreground)
        return loss

    def period_loss(self, pred, target, mask):
	pred = pred[mask==1]
	target = target[mask==1]
        loss = self.MSELoss(pred, target)
        return loss

    def seg_acc(self, pred, target):
	n, h, w = target.size()
	label = pred.data.cpu().numpy().argmax(1)
	target = target.data.cpu().numpy()
	acc = np.sum(label == target) / float(n * h * w)
	return acc
	
