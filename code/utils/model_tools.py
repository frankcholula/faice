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


def freeze_layers(model, freeze_until_layer):
    """
    Freeze layers until the specified layer index.
    """
    layers = 0
    for name, param in model.named_parameters():
        # Split the parameter name by '.'
        parts = name.split(".")

        # Check if the second part is a digit (e.g., '0', '1')
        if len(parts) > 1 and parts[1].isdigit():
            layer_index = int(parts[1])
            layers += 1
            if layer_index < freeze_until_layer:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            # Skip parameters that do not match the expected format
            continue
    print(
        f"The model has {layers} layers and freeze the front {freeze_until_layer} layers"
    )
