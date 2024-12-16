import os
import sys
from frontier_exploration.models.network import FrontierNetworkModel
from frontier_exploration.models.trainer import FrontierTrainer
from frontier_exploration.environment.robot import Robot
from frontier_exploration.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR

def main():
    # 创建模型
    model = FrontierNetworkModel(
        input_shape=MODEL_CONFIG['input_shape'],
        max_frontiers=MODEL_CONFIG['max_frontiers']
    )
    
    # 创建机器人环境
    robot = Robot(0, train=True, plot=True)
    
    # 创建训练器
    trainer = FrontierTrainer(
        model=model,
        robot=robot,
        memory_size=MODEL_CONFIG['memory_size'],
        batch_size=MODEL_CONFIG['batch_size'],
        gamma=MODEL_CONFIG['gamma']
    )
    
    # 开始训练
    trainer.train(
        episodes=TRAIN_CONFIG['episodes'],
        steps_per_episode=TRAIN_CONFIG['steps_per_episode'],
        target_update_freq=TRAIN_CONFIG['target_update_freq']
    )
    
    # 保存模型
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, 'frontier_model_final.h5'))

if __name__ == '__main__':
    main()