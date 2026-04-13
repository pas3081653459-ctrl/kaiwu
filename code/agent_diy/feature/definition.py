#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
from common_python.utils.common_func import create_cls, attached
from agent_diy.conf.conf import Config

ObsData = create_cls("ObsData", feature=None, legal_action=None)
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)

def sample_process(list_sample_data):
    """
    负责整理采样得到的对局片段数据(Trajectories), 为最后的 PPO 步骤计算时间差分优势与回报积累
    """
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data

def _calc_gae(list_sample_data):
    """
    Generalized Advantage Estimation 泛化优势函数评估。
    从后向前累加，以衰减系数平滑真实奖励与模型预期奖励（Critic预测）的偏差。
    """
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        delta = -sample.value + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.reward_sum = gae + sample.value

def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):
    """
    Reward shaping calculation. Since we compute rewards intrinsically in the 
    preprocessor feature_process, we can just pass here.
    """
    pass
