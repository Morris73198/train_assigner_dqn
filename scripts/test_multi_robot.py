import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from two_robot_exploration.models.multi_robot_network import MultiRobotNetworkModel
from two_robot_exploration.environment.multi_robot import Robot
from two_robot_exploration.config import MODEL_CONFIG, MODEL_DIR

def test_model(model_path, num_episodes=5, plot=True):
    """
    測試訓練好的多機器人探索模型
    
    Args:
        model_path: 模型檔案路徑(.h5檔案)
        num_episodes: 測試輪數
        plot: 是否顯示可視化
    """
    try:
        # 載入模型
        print("Loading model from:", model_path)
        model = MultiRobotNetworkModel(
            input_shape=MODEL_CONFIG['input_shape'],
            max_frontiers=MODEL_CONFIG['max_frontiers']
        )
        model.load(model_path)
        
        # 創建測試環境
        print("Initializing test environment...")
        robot1, robot2 = Robot.create_shared_robots(
            index_map=0,
            train=False,  # 使用測試地圖
            plot=plot
        )
        
        # 測試統計
        episode_stats = {
            'exploration_progress': [],
            'steps': [],
            'robot1_path_length': [],
            'robot2_path_length': [],
            'completion_time': []
        }
        
        for episode in range(num_episodes):
            print(f"\nStarting episode {episode + 1}/{num_episodes}")
            
            # 重置環境
            state = robot1.begin()
            robot2.begin()
            
            steps = 0
            robot1_path_length = 0
            robot2_path_length = 0
            
            while not (robot1.check_done() or robot2.check_done()):
                # 獲取當前狀態
                frontiers = robot1.get_frontiers()
                if len(frontiers) == 0:
                    break
                    
                robot1_pos = robot1.get_normalized_position()
                robot2_pos = robot2.get_normalized_position()
                
                # 使用模型預測動作
                state_batch = np.expand_dims(state, 0)
                frontiers_batch = np.expand_dims(
                    robot1.pad_frontiers(frontiers) 
                    if hasattr(robot1, 'pad_frontiers') 
                    else np.zeros((50, 2)), 0
                )
                robot1_pos_batch = np.expand_dims(robot1_pos, 0)
                robot2_pos_batch = np.expand_dims(robot2_pos, 0)
                
                predictions = model.predict(
                    state_batch, frontiers_batch,
                    robot1_pos_batch, robot2_pos_batch
                )
                
                # 選擇動作
                valid_frontiers = min(MODEL_CONFIG['max_frontiers'], len(frontiers))
                robot1_action = np.argmax(predictions['robot1'][0, :valid_frontiers])
                robot2_q = predictions['robot2'][0, :valid_frontiers].copy()
                robot2_q[robot1_action] *= 0.5  # 降低已被Robot1選擇的目標權重
                robot2_action = np.argmax(robot2_q)
                
                # 執行動作
                robot1_target = frontiers[robot1_action]
                robot2_target = frontiers[robot2_action]
                
                # Robot 1移動
                next_state1, r1, d1 = robot1.move_to_frontier(robot1_target)
                robot1_path_length += np.linalg.norm(
                    robot1.robot_position - robot1.old_position
                )
                
                # 更新地圖
                robot2.op_map = robot1.op_map.copy()
                
                # Robot 2移動
                next_state2, r2, d2 = robot2.move_to_frontier(robot2_target)
                robot2_path_length += np.linalg.norm(
                    robot2.robot_position - robot2.old_position
                )
                
                # 確保地圖同步
                robot1.op_map = robot2.op_map.copy()
                
                # 更新機器人位置信息
                robot1.other_robot_position = robot2.robot_position.copy()
                robot2.other_robot_position = robot1.robot_position.copy()
                
                state = next_state1
                steps += 1
                
                # 更新可視化
                if plot and steps % 10 == 0:  # 每10步更新一次顯示
                    robot1.plot_env()
                    robot2.plot_env()
                    plt.pause(0.001)  # 短暫暫停以更新顯示
            
            # 記錄episode統計
            final_progress = robot1.get_exploration_progress()
            episode_stats['exploration_progress'].append(final_progress)
            episode_stats['steps'].append(steps)
            episode_stats['robot1_path_length'].append(robot1_path_length)
            episode_stats['robot2_path_length'].append(robot2_path_length)
            
            print(f"Episode {episode + 1} Results:")
            print(f"Steps taken: {steps}")
            print(f"Final exploration progress: {final_progress:.1%}")
            print(f"Robot1 path length: {robot1_path_length:.2f}")
            print(f"Robot2 path length: {robot2_path_length:.2f}")
            
            # 重置環境
            state = robot1.reset()
            robot2.reset()
        
        # 顯示總體統計
        print("\nOverall Test Results:")
        print(f"Average steps: {np.mean(episode_stats['steps']):.2f}")
        print(f"Average exploration progress: {np.mean(episode_stats['exploration_progress']):.1%}")
        print(f"Average Robot1 path length: {np.mean(episode_stats['robot1_path_length']):.2f}")
        print(f"Average Robot2 path length: {np.mean(episode_stats['robot2_path_length']):.2f}")
        
        # 繪製測試結果
        plot_test_results(episode_stats)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理資源
        if plot:
            if hasattr(robot1, 'cleanup_visualization'):
                robot1.cleanup_visualization()
            if hasattr(robot2, 'cleanup_visualization'):
                robot2.cleanup_visualization()

def plot_test_results(stats):
    """繪製測試結果統計圖"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    episodes = range(1, len(stats['steps']) + 1)
    
    # 探索進度
    axs[0, 0].plot(episodes, stats['exploration_progress'], 'b-', marker='o')
    axs[0, 0].set_title('Exploration Progress')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Progress (%)')
    axs[0, 0].grid(True)
    
    # 步數
    axs[0, 1].plot(episodes, stats['steps'], 'r-', marker='o')
    axs[0, 1].set_title('Steps per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Steps')
    axs[0, 1].grid(True)
    
    # 路徑長度比較
    axs[1, 0].plot(episodes, stats['robot1_path_length'], 'purple', 
                   marker='o', label='Robot1')
    axs[1, 0].plot(episodes, stats['robot2_path_length'], 'orange', 
                   marker='o', label='Robot2')
    axs[1, 0].set_title('Path Length Comparison')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Path Length')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # 總路徑長度
    total_path_length = np.array(stats['robot1_path_length']) + \
                       np.array(stats['robot2_path_length'])
    axs[1, 1].plot(episodes, total_path_length, 'g-', marker='o')
    axs[1, 1].set_title('Total Path Length')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Length')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()

def main():
    # 指定要使用的模型文件
    model_file = 'multi_robot_model_ep000120.h5'
    model_path = os.path.join(MODEL_DIR, model_file)
    
    # 檢查文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        print(f"Available files in {MODEL_DIR}:")
        for file in os.listdir(MODEL_DIR):
            print(f"  - {file}")
        return
        
    print(f"\nUsing model: {model_file}")
    print(f"Model path: {model_path}")
    
    # 設定默認的測試輪數
    num_episodes = 5
    test_model(model_path, num_episodes=num_episodes, plot=True)

if __name__ == '__main__':
    main()