#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Author: Tencent AI Arena Authors

自定义的数据预处理模块，负责：
1. 解析当前状态信息（包含自身，怪物，可通行地图）
2. 特征工程处理（距离目标与奖励计算）
3. 状态组装：打包为一维特征向量并设定当帧的Reward奖惩反馈
"""

import numpy as np

# 预定义的基准常量用于将真实数值范围归一化
MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_DIST_BUCKET = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0

def _norm(v, v_max, v_min=0.0):
    """
    通用归一化函数，将数值裁剪并线性映射到 [0, 1] 范围内。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0

class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        # 内部状态：记录步数及跟踪变化指标（用于奖励计算）
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.last_treasure_count = 0
        self.last_buff_count = 0
        self.last_hero_pos = None
        self.pos_history = []  # 用于检测是否在原地打转或反复徘徊
        self.last_closest_treasure_dist = 1.0

    def feature_process(self, env_obs, last_action):
        """
        核心处理入口，从环境原始字典中提取有效信息：自身状态、视角内怪物、局部墙体以及各种可拾取物。
        返回特征(feature)、合法掩码(legal_action) 以及当前帧即时回报(reward)
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        # ---------------------------------------------
        # 1. 英雄本体特征 (4维)
        # ---------------------------------------------
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hx, hz = hero_pos["x"], hero_pos["z"]
        
        hero_x_norm = _norm(hx, MAP_SIZE)
        hero_z_norm = _norm(hz, MAP_SIZE)
        flash_cd_norm = _norm(hero.get("flash_cooldown", 0), MAX_FLASH_CD)
        buff_remain_norm = _norm(hero.get("buff_remaining_time", 0), MAX_BUFF_DURATION)
        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # ---------------------------------------------
        # 2. 怪物特征 (5维 * 2 = 10维)
        # 融合视野判断：视野内使用精确坐标差，视野外兜底使用官方方向和距离桶
        # 特征：[是否存在, 单位化相对dx, 单位化相对dz, 速度归一化, 统一距离归一化]
        # ---------------------------------------------
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        
        # 相对英雄方位映射 (0=无效,1=东,2=东北,3=北,4=西北,5=西,6=西南,7=南,8=东南)
        # x向右为正，z向下为正(南)
        dir_map = {
            0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, -1.0), 3: (0.0, -1.0),
            4: (-1.0, -1.0), 5: (-1.0, 0.0), 6: (-1.0, 1.0), 7: (0.0, 1.0), 8: (1.0, 1.0)
        }
        
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_exist = 1.0
                m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)
                
                m_pos = m.get("pos")
                m_dir = m.get("hero_relative_direction", 0)
                bucket = m.get("hero_l2_distance", 5)
                
                # 先明确判断 pos 里面有没有具体数据（在视野外时，环境可能不传 pos 或传 None）
                if m_pos is not None and "x" in m_pos and "z" in m_pos:
                    # 视野内：提取精确坐标并计算精准差值
                    mx, mz = m_pos["x"], m_pos["z"]
                    ex_dx = mx - hx
                    ex_dz = mz - hz
                    exact_dist = np.sqrt(ex_dx**2 + ex_dz**2)
                    
                    if exact_dist > 0:
                        dx = ex_dx / exact_dist
                        dz = ex_dz / exact_dist
                    else:
                        dx, dz = 0.0, 0.0
                    # 精确距离标度 (0 ~ 1.0)
                    dist_norm = exact_dist / (MAP_SIZE * 1.41)
                else:
                    # 视野外：pos 不可用，使用官方提供的方向枚举与距离桶
                    bx, bz = dir_map.get(m_dir, (0.0, 0.0))
                    blen = np.sqrt(bx**2 + bz**2)
                    if blen > 0:
                        dx, dz = bx / blen, bz / blen  # 化为单位向量
                    else:
                        dx, dz = 0.0, 0.0
                    
                    # 桶为: 0=[0,30), 1=[30,60)... 估算真实物理中心距离进行平滑兜底
                    est_dist = bucket * 30.0 + 15.0
                    dist_norm = min(1.0, est_dist / (MAP_SIZE * 1.41))
                
                monster_feats.append(np.array([is_exist, dx, dz, m_speed_norm, dist_norm], dtype=np.float32))
            else:
                # 怪物未生成时，设为最远、无方向
                monster_feats.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        # ---------------------------------------------
        # 3. 局部地图特征 (9x9 = 81维)
        # 获取英雄周边可行走的道路特征 (值为 1 代表可通行， 0 代表障碍物)
        # ---------------------------------------------
        map_feat = np.zeros(81, dtype=np.float32)
        if map_info is not None and len(map_info) >= 21:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 4, center + 5):
                for col in range(center - 4, center + 5):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        # ---------------------------------------------
        # 4. 资源/增益特征 (宝箱 9维， Buff 6维)
        # 先搜集，后基于与英雄的欧氏距离排序提取距离最近的多个对象
        # ---------------------------------------------
        organs = frame_state.get("organs", [])
        treasures = []
        buffs = []
        for o in organs:
            if o["status"] == 1:
                ox, oz = o["pos"]["x"], o["pos"]["z"]
                dist = np.sqrt((hx - ox)**2 + (hz - oz)**2)
                if o["sub_type"] == 1:
                    treasures.append((dist, ox, oz))
                elif o["sub_type"] == 2:
                    buffs.append((dist, ox, oz))
        
        treasures.sort(key=lambda x: x[0])
        buffs.sort(key=lambda x: x[0])
        
        treasure_feat = []
        for i in range(3):
            if i < len(treasures):
                d, ox, oz = treasures[i]
                treasure_feat.extend([_norm(ox - hx, MAP_SIZE, -MAP_SIZE), _norm(oz - hz, MAP_SIZE, -MAP_SIZE), _norm(d, MAP_SIZE * 1.41)])
            else:
                treasure_feat.extend([0.0, 0.0, 1.0])
                
        buff_feat = []
        for i in range(2):
            if i < len(buffs):
                d, ox, oz = buffs[i]
                buff_feat.extend([_norm(ox - hx, MAP_SIZE, -MAP_SIZE), _norm(oz - hz, MAP_SIZE, -MAP_SIZE), _norm(d, MAP_SIZE * 1.41)])
            else:
                buff_feat.extend([0.0, 0.0, 1.0])

        # ---------------------------------------------
        # 5. 合法运动/闪现掩码处理 (16维)
        # 控制非法（越界/墙壁上/技能未冷却）动作输出
        # ---------------------------------------------
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 16}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]

        if sum(legal_action) == 0:
            legal_action = [1] * 16

        # ---------------------------------------------
        # 5.5 闪现落点安全评估特征 (8维)
        # 预测8个闪现方向的落点距离比例，帮助模型判断闪现过后会不会撞墙
        # 闪现距离：直线10，斜向8
        # 方向：右(0), 右上(1), 上(2), 左上(3), 左(4), 左下(5), 下(6), 右下(7)
        # 由于网格z轴对应通常的上下，我们设定射线步进
        # ---------------------------------------------
        flash_feat = np.zeros(8, dtype=np.float32)
        if map_info is not None and len(map_info) >= 21:
            center = len(map_info) // 2
            # (dx, dy, max_step) 注: 这里dy是row偏移, dx是col偏移
            # 下方表示为通常屏幕坐标系(y正向为下,x正向为右),这里只需保持与方向映射一致即可
            dirs = [
                (1, 0, 10),   # 0: 右
                (1, -1, 8),   # 1: 右上
                (0, -1, 10),  # 2: 上
                (-1, -1, 8),  # 3: 左上
                (-1, 0, 10),  # 4: 左
                (-1, 1, 8),   # 5: 左下
                (0, 1, 10),   # 6: 下
                (1, 1, 8)     # 7: 右下
            ]
            
            for i, (dx, dy, m_step) in enumerate(dirs):
                actual_step = 0
                # 检查落点，如果在地图内且无障碍，则距离为满m_step
                ty, tx = center + dy * m_step, center + dx * m_step
                if 0 <= ty < len(map_info) and 0 <= tx < len(map_info[0]) and map_info[ty][tx] != 0:
                    actual_step = m_step
                else:
                    # 否则，从最远端往回退，找到最后一个安全格子
                    for step in range(m_step - 1, -1, -1):
                        cy, cx = center + dy * step, center + dx * step
                        if 0 <= cy < len(map_info) and 0 <= cx < len(map_info[0]) and map_info[cy][cx] != 0:
                            actual_step = step
                            break
                            
                flash_feat[i] = _norm(actual_step, m_step)  # 归一化到 [0, 1]

        # ---------------------------------------------
        # 5.6 动作记录与卡死标志 (17维)
        # 上一帧动作的 one-hot 编码 (16) + 是否陷入卡死状态 (1)
        # ---------------------------------------------
        last_action_feat = np.zeros(16, dtype=np.float32)
        if 0 <= last_action < 16:
            last_action_feat[last_action] = 1.0

        self.pos_history.append((hx, hz))
        is_stuck = 0.0
        if len(self.pos_history) > 10:
            self.pos_history.pop(0)
            xs = [p[0] for p in self.pos_history]
            zs = [p[1] for p in self.pos_history]
            if (max(xs) - min(xs)) <= 2.0 and (max(zs) - min(zs)) <= 2.0:
                is_stuck = 1.0
        stuck_feat = np.array([is_stuck], dtype=np.float32)

        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # 全局拼贴
        feature = np.concatenate([
            hero_feat,
            monster_feats[0],
            monster_feats[1],
            map_feat,
            np.array(legal_action, dtype=np.float32),
            progress_feat,
            np.array(treasure_feat, dtype=np.float32),
            np.array(buff_feat, dtype=np.float32),
            flash_feat,
            last_action_feat,
            stuck_feat
        ])

        # ---------------------------------------------
        # 6. Reward (奖励反馈)
        # 包括基础存活经验分、与怪物的危险空间距离奖惩（防过近），
        # 取拾到金币增益等动作时的稀疏奖励以及碰墙惩罚(避免模型因怕死无限贴墙发呆)
        # ---------------------------------------------
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])

        survive_reward = 0.01
        
        # 躲避怪兽的 Reward：相比上回合距离变远了就会给正分，被怪物拉近了就会扣分
        # cur_min_dist_norm 越大（距离越远）越好。
        dist_shaping = 10.0 * (cur_min_dist_norm - self.last_min_monster_dist_norm)
        
        # 极近距离惩罚：如果跟怪物距离过近（约小于视阈的15%），持续扣分逼迫其学习闪现和逃跑
        close_penalty = 0.0
        if cur_min_dist_norm < 0.15:
            close_penalty = -0.05

        cur_treasure_count = env_info.get("treasures_collected", 0)
        treasure_reward = 1.0 if cur_treasure_count > self.last_treasure_count else 0.0
        
        # 增加朝向宝箱探索的引力（指引方向，防止在没有怪物时乱晃）
        cur_closest_treasure_dist = 1.0 # 默认为最大距离
        if len(treasures) > 0:
            # treasures里的记录是真实距离
            cur_closest_treasure_dist = _norm(treasures[0][0], MAP_SIZE * 1.41)
            
        treasure_shaping = 0.0
        if self.step_no > 0: # 避免第一帧的异常shaping
            treasure_shaping = 1.0 * (self.last_closest_treasure_dist - cur_closest_treasure_dist)
            
        cur_buff_count = env_info.get("collected_buff", 0)
        buff_reward = 0.5 if cur_buff_count > self.last_buff_count else 0.0

        wall_penalty = 0.0
        flash_reward = 0.0
        stuck_penalty = 0.0
        
        # 使用刚刚计算的卡死特权，给予重度惩罚
        if is_stuck > 0:
            stuck_penalty = -0.05
                
        if self.last_hero_pos is not None and last_action >= 0:
            old_hx, old_hz = self.last_hero_pos
            # 无论走路还是闪现，只要没动，说明撞墙或者被挡住了
            if hx == old_hx and hz == old_hz:
                wall_penalty = -0.1
            
            # 如果动作是闪现(8-15)且成功拉开了与怪物的距离，给一次性的大奖励，鼓励用闪现逃跑或穿墙
            if 8 <= last_action <= 15 and (cur_min_dist_norm - self.last_min_monster_dist_norm) > 0.01:
                flash_reward = 0.5

        self.last_hero_pos = (hx, hz)
        self.last_treasure_count = cur_treasure_count
        self.last_buff_count = cur_buff_count
        self.last_min_monster_dist_norm = cur_min_dist_norm
        self.last_closest_treasure_dist = cur_closest_treasure_dist

        reward = [survive_reward + dist_shaping + close_penalty + treasure_reward + buff_reward + wall_penalty + flash_reward + treasure_shaping + stuck_penalty]

        return feature, legal_action, reward