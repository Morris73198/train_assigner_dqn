import os
from frontier_exploration.models.network import FrontierNetworkModel
from frontier_exploration.environment.robot import Robot
from frontier_exploration.config import MODEL_CONFIG, MODEL_DIR
import numpy as np

def main():
    # 加载模型
    model = FrontierNetworkModel(
        input_shape=MODEL_CONFIG['input_shape'],
        max_frontiers=MODEL_CONFIG['max_frontiers']
    )
    model.load(os.path.join(MODEL_DIR, 'frontier_model_final.h5'))
    
    # 创建测试环境
    robot = Robot(0, train=False, plot=True)
    
    # 开始测试
    state = robot.begin()
    total_reward = 0
    steps = 0
    
    while True:
        # 获取frontiers
        frontiers = robot.get_frontiers()
        if len(frontiers) == 0:
            break
            
        # 预测最佳frontier
        q_values = model.predict(
            np.expand_dims(state, 0),
            np.expand_dims(frontiers, 0)
        )[0]
        action = np.argmax(q_values[:len(frontiers)])
        selected_frontier = frontiers[action]
        
        # 移动到选定的frontier
        next_state, reward, done = robot.move_to_frontier(selected_frontier)
        
        total_reward += reward
        steps += 1
        state = next_state
        
        if done:
            break
    
    print(f"Test completed:")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward:.2f}")

if __name__ == '__main__':
    main()