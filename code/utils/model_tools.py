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
    ema_params = OrderedDict(ema_model.parameters())
    model_params = OrderedDict(model.parameters())

    for name, param in model_params.items():
        # TO-DO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    # for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
    #     ema_param.copy_(decay * ema_param + (1 - decay) * model_param)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
