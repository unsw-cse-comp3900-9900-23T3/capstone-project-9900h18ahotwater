import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

def train_one_epoch(model, optimizer, data_loader, device, epoch, threshold=0.5):
    model.train()
    loss_function = torch.nn.BCELoss()
    accu_loss = torch.zeros(1).to(device)  # accumulative loss
    tp = torch.zeros(1).to(device)  # accumulative tp
    fn = torch.zeros(1).to(device)  # accumulative fn
    fp = torch.zeros(1).to(device)  # accumulative fp
    tn = torch.zeros(1).to(device)  # accumulative tn
    optimizer.zero_grad()

    # sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        x, y = data
        # sample_num += y.shape[0]*y.shape[1]

        pred = model(x.to(device))
        # print(pred)
        # print(pred.shape)
        # print(y)
        pred_class = torch.where(pred > threshold, torch.ones_like(pred), torch.zeros_like(pred))
        tp += torch.sum(torch.mul(pred_class,y))
        tn += torch.sum(torch.mul(1-pred_class,1-y))
        fn += torch.sum(torch.mul(1-pred_class,y))
        fp += torch.sum(torch.mul(pred_class,1-y))
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*precision*recall/(precision+recall)
        # accu_num += torch.eq(pred_class, y.to(device)).sum()
        loss = loss_function(pred, y.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               precision.item(),
                                                                               recall.item(),
                                                                               f1.item())

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), precision.item(), recall.item(), f1.item()


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, threshold=0.5):
    loss_function = torch.nn.BCELoss()

    model.eval()

    
    accu_loss = torch.zeros(1).to(device)  # accumulative loss
    tp = torch.zeros(1).to(device)  # accumulative tp
    fn = torch.zeros(1).to(device)  # accumulative fn
    fp = torch.zeros(1).to(device)  # accumulative fp
    tn = torch.zeros(1).to(device)  # accumulative tn

    # sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        x, y = data
        # sample_num += x.shape[0]

        pred = model(x.to(device))
        pred_class = torch.where(pred > threshold, torch.ones_like(pred), torch.zeros_like(pred))
        tp += torch.sum(torch.mul(pred_class,y))
        tn += torch.sum(torch.mul(1-pred_class,1-y))
        fn += torch.sum(torch.mul(1-pred_class,y))
        fp += torch.sum(torch.mul(pred_class,1-y))
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*precision*recall/(precision+recall)
        # accu_num += torch.eq(pred_class, y.to(device)).sum()

        loss = loss_function(pred, y.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               precision.item(),
                                                                               recall.item(),
                                                                               f1.item())

    return accu_loss.item() / (step + 1), precision.item(), recall.item(), f1.item()