# -*- coding: UTF-8 -*-
"""
@Time : 03/05/2025 15:07
@Author : xiaoguangliang
@File : model_tools.py
@Project : code
"""
from collections import OrderedDict

import torch
import pandas as pd


def name_to_label(name):
    train_label_path = 'datasets/celeba_hq_split/celebaAHQ_train.xlsx'
    label_data = pd.read_excel(train_label_path)
    label_dict = dict(zip(label_data['image'], label_data['label']))
    name = int(name)
    return label_dict[name]


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = ema_model.parameters()
    model_params = model.parameters()

    for targ, src in zip(ema_params, model_params):
        targ.detach().mul_(decay).add_(src, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
