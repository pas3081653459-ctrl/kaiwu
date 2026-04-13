#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch
from agent_diy.conf.conf import Config
import os
import time

class Algorithm:
    """
    计算PPO系列损失函数并基于SampleData去执行反向传播迭代的网络优化算法类。
    利用开悟 Learner 进行自动并行调用：
        Loss = ValueLoss * VF_Coef + PolicyLoss - Entropy * BETA
    """
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        # 获取在conf下的约束范围配置
        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        """
        Learner 获取到一批带有时序或打乱的优势计算后（GAE完了之后）的观测进行深度学习训练。
        """
        # 数据转张量(tensor)并放上对应的主机端(CPU\GPU)
        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).to(self.device)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device)

        # 设置为训练模式以开启相关节点的Dropout和反向状态
        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)

        total_loss, info_list = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        total_loss.backward()
        # 将参数使用指定范围进行渐变裁剪，使更新较为稳定，不易在优势估计高偏时发散
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info_list[0].item(), 4),
                "policy_loss": round(info_list[1].item(), 4),
                "entropy_loss": round(info_list[2].item(), 4),
                "reward": round(reward.mean().item(), 4),
            }
            self.logger.info(
                f"[train] total_loss:{results['total_loss']} policy_loss:{results['policy_loss']} value_loss:{results['value_loss']} entropy:{results['entropy_loss']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_loss(self, logits, value_pred, legal_action, old_action, old_prob, advantage, old_value, reward_sum, reward):
        """
        基于截断的目标函数，防止更新步长过大导致的剧烈震荡
        """
        prob_dist = self._masked_softmax(logits, legal_action)

        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        adv = advantage.view(-1, 1)
        
        # Clip操作截断大比率，保守策略优化
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        # Critic网络预测回归的均方差与限幅(Value Loss)
        vp = value_pred
        ov = old_value
        tdret = reward_sum
        value_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = 0.5 * torch.maximum(torch.square(tdret - vp), torch.square(tdret - value_clip)).mean()

        # 计算熵，用于在训练早中期鼓励探索(Entropy Loss)
        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()

        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        return total_loss, [value_loss, policy_loss, entropy_loss]

    def _masked_softmax(self, logits, legal_action):
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(label, dim=1)
