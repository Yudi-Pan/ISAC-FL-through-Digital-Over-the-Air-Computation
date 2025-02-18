#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import random

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset

class LocalUpdate(object):
    def __init__(self, args, batch_size, dataset, datasize):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        total_samples = list(range(len(dataset)))
        random_samples = random.sample(total_samples, datasize)
        sub_dataset = torch.utils.data.Subset(dataset, random_samples)
        self.ldr_train = DataLoader(sub_dataset, batch_size=batch_size, shuffle=True)

    def train_batch(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        old_state_dict = copy.deepcopy(net.state_dict())
        epoch_loss = []
        for it in range(1):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # 求取状态字典的差异（仅记录参数值的变化）
        state_dict_diff = {}
        for name, param in net.state_dict().items():
            # if name in old_state_dict and not torch.equal(param, old_state_dict[name]):
            state_dict_diff[name] = param - old_state_dict[name]
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), state_dict_diff
