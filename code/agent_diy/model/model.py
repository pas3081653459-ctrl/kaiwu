#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


from agent_diy.conf.conf import Config

def make_fc_layer(in_features, out_features):
    """
    创建一个带正交初始化的线性全连接层，防止由于权重过大产生的梯度爆炸。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc

class Model(nn.Module):
    """
    PPO 网络模型：单 MLP 骨干结构（共享特征提取）+ Actor（动作策略输出头）/ Critic（状态价值输出头）
    """
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_diy"
        self.device = device
        
        # 读取配置中的空间维度参数
        input_dim = Config.DIM_OF_OBSERVATION
        hidden_dim = 256
        mid_dim = 128
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # 共享特征提取网络
        self.backbone = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
        )

        # 策略头和价值头
        self.actor_head = make_fc_layer(mid_dim, action_num)
        self.critic_head = make_fc_layer(mid_dim, value_num)

    def forward(self, obs, inference=False):
        hidden = self.backbone(obs)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
