import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')

# 模型配置
MODEL_CONFIG = {
    'input_shape': (84, 84, 1),
    'max_frontiers': 200,  # 这个值要和动作空间大小匹配
    'memory_size': 10000,
    'batch_size': 32,
    'gamma': 0.99,
    'epsilon_min': 0.1,
    'epsilon_decay': 0.995
}

# 训练配置
TRAIN_CONFIG = {
    'episodes': 1000000,
    'steps_per_episode': 5000,
    'target_update_freq': 10,
    'save_freq': 100
}

# 机器人配置
ROBOT_CONFIG = {
    # 基础配置
    'sensor_range': 80,
    'robot_size': 6,
    'local_size': 40,
    'finish_percent': 0.985,
    
    # 移动优化配置
    'movement_step': 3,         # 基本移动步长
    'max_frontier_skip': 2,      # 每次跳过的路径点数
    'path_simplification': 0.1,  # 路径简化阈值
    'min_frontier_dist': 30,     # 最小frontier选择距离
    'target_reach_threshold': 5, # 目标达到阈值
    
    # A*寻路优化
    'safety_distance': 30,        # 安全距离
    'diagonal_weight': 1.2,      # 对角线移动代价
    
    # 可视化更新频率
    'plot_interval': 2,          # 更新间隔
}

# 奖励函数配置
REWARD_CONFIG = {
    'exploration_weight': 2.0,     # 探索奖励权重
    'movement_penalty': -0.005,    # 移动惩罚
    'collision_penalty': -0.5,     # 碰撞惩罚
    'target_completion_reward': 5.0, # 到达目标奖励
    'completion_reward': 10.0       # 完成探索奖励
}