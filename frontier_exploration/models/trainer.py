import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
from frontier_exploration.config import MODEL_DIR

class FrontierTrainer:
    def __init__(self, model, robot, memory_size=10000, batch_size=16, gamma=0.99):
        """
        初始化训练器
        
        Args:
            model: FrontierNetworkModel实例
            robot: Robot实例
            memory_size: 经验回放缓冲区大小
            batch_size: 训练批次大小
            gamma: 奖励折扣因子
        """
        self.model = model
        self.robot = robot
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        # 训练参数
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.1  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        
        # 训练历史记录
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'exploration_rates': [],
            'losses': []
        }
    
    def remember(self, state, frontiers, action, reward, next_state, next_frontiers, done):
        """
        将经验存储到回放缓冲区
        """
        self.memory.append((state, frontiers, action, reward, next_state, next_frontiers, done))
    
    def pad_frontiers(self, frontiers):
        """
        将frontiers填充到固定长度
        
        Args:
            frontiers: 原始frontiers列表
            
        Returns:
            填充后的frontiers数组
        """
        padded = np.zeros((self.model.max_frontiers, 2))
        if len(frontiers) > 0:
            n_frontiers = min(len(frontiers), self.model.max_frontiers)
            padded[:n_frontiers] = frontiers[:n_frontiers]
        return padded
    
    def choose_action(self, state, frontiers):
        """选择动作（frontier）"""
        # 如果没有frontiers，说明地图探索完成，需要重置环境
        if len(frontiers) == 0:
            # 重置环境并获取新的状态和frontiers
            state = self.robot.reset()
            frontiers = self.robot.get_frontiers()
            
            # 如果新地图也没有frontiers（极少发生），返回0
            if len(frontiers) == 0:
                return 0
        
        # epsilon-greedy策略
        if np.random.random() < self.epsilon:
            return np.random.randint(min(20, len(frontiers)))
        
        # 使用模型预测
        state_batch = np.expand_dims(state, 0)
        frontiers_batch = np.expand_dims(self.pad_frontiers(frontiers), 0)
        
        # 修正：使用正确的模型预测方法
        q_values = self.model.predict(state_batch, frontiers_batch)
        
        # 只考虑有效的frontier数量
        valid_q = q_values[0, :len(frontiers)]
        return np.argmax(valid_q)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        states = []
        frontiers_batch = []
        next_states = []
        next_frontiers_batch = []
        
        for state, frontiers, action, reward, next_state, next_frontiers, done in batch:
            if len(state.shape) == 2:
                state = np.expand_dims(state, axis=-1)
            if len(next_state.shape) == 2:
                next_state = np.expand_dims(next_state, axis=-1)
                
            action = min(action, 19)
                
            states.append(state)
            if len(frontiers) == 0:
                frontiers = np.zeros((1, 2))
            frontiers_batch.append(self.pad_frontiers(frontiers))
            
            next_states.append(next_state)
            if len(next_frontiers) == 0:
                next_frontiers = np.zeros((1, 2))
            next_frontiers_batch.append(self.pad_frontiers(next_frontiers))
        
        states = np.array(states)
        frontiers_batch = np.array(frontiers_batch)
        next_states = np.array(next_states)
        next_frontiers_batch = np.array(next_frontiers_batch)
        
        # 将预测转换为NumPy数组
        target_q = self.model.target_model({'map_input': next_states, 
                                        'frontier_input': next_frontiers_batch}).numpy()
        current_q = self.model.model({'map_input': states, 
                                    'frontier_input': frontiers_batch}).numpy()
        
        # 使用NumPy数组更新Q值
        for i, (_, _, action, reward, _, _, done) in enumerate(batch):
            action = min(action, 19)
            if done:
                current_q[i][action] = reward
            else:
                current_q[i][action] = reward + self.gamma * np.max(target_q[i])
        
        # 训练模型
        loss = self.model.model.train_on_batch(
            x={'map_input': states, 'frontier_input': frontiers_batch},
            y=current_q
        )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def train(self, episodes=1000000, target_update_freq=10, save_freq=10):
        """
        训练过程
        
        Args:
            episodes: 训练轮数(地图数)
            target_update_freq: 目标网络更新频率
            save_freq: 保存检查点频率
        """
        for episode in range(episodes):
            state = self.robot.begin()  # 开始新地图
            total_reward = 0
            steps = 0
            episode_losses = []
            
            # 在这个地图上训练直到完成为止
            while not self.robot.check_done():  # 使用check_done检查地图是否真正探索完成
                frontiers = self.robot.get_frontiers()
                if len(frontiers) == 0:
                    break  # 无可用的frontier点
                    
                action = self.choose_action(state, frontiers)
                selected_frontier = frontiers[action]
                
                # 移动到选定的frontier
                next_state, reward, move_done = self.robot.move_to_frontier(selected_frontier)
                next_frontiers = self.robot.get_frontiers()
                
                # 存储经验并训练
                self.remember(state, frontiers, action, reward, next_state, next_frontiers, move_done)
                loss = self.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                # 如果这次移动失败了，继续尝试其他frontier
                if move_done and not self.robot.check_done():
                    continue
            
            # 更新训练历史
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['episode_lengths'].append(steps)
            self.training_history['exploration_rates'].append(self.epsilon)
            self.training_history['losses'].append(np.mean(episode_losses) if episode_losses else 0)
            
            # 更新目标网络
            if (episode + 1) % target_update_freq == 0:
                self.model.update_target_model()
            
            # 保存检查点
            if (episode + 1) % save_freq == 0:
                self.save_checkpoint(episode + 1)
            
            # 打印训练信息
            exploration_progress = self.robot.get_exploration_progress()
            print(f"\nEpisode {episode + 1}/{episodes} (Map {self.robot.li_map})")
            print(f"Steps: {steps}, Total Reward: {total_reward:.2f}")
            print(f"Epsilon: {self.epsilon:.3f}")
            print(f"Average Loss: {self.training_history['losses'][-1]:.6f}")
            print(f"Exploration Progress: {exploration_progress:.1%}")
            
            if exploration_progress >= self.robot.finish_percent:
                print("Map fully explored!")
            else:
                print("Map exploration incomplete")
            print("-" * 50)
            
            # 准备下一个地图
            state = self.robot.reset()
        
        # 训练结束后保存最终模型
        self.save_checkpoint(episodes)
    
    def plot_training_progress(self):
        """
        绘制训练进度图
        """
        fig, axs = plt.subplots(4, 1, figsize=(10, 15))
        
        # 绘制奖励
        episodes = range(1, len(self.training_history['episode_rewards']) + 1)
        axs[0].plot(episodes, self.training_history['episode_rewards'])
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')
        
        # 绘制步数
        axs[1].plot(episodes, self.training_history['episode_lengths'])
        axs[1].set_title('Episode Lengths')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Steps')
        
        # 绘制探索率
        axs[2].plot(episodes, self.training_history['exploration_rates'])
        axs[2].set_title('Exploration Rate')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Epsilon')
        
        # 绘制损失
        axs[3].plot(episodes, self.training_history['losses'])
        axs[3].set_title('Training Loss')
        axs[3].set_xlabel('Episode')
        axs[3].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()
    
    def save_training_history(self, filename='training_history.npz'):
        """
        保存训练历史
        """
        np.savez(filename, 
                episode_rewards=self.training_history['episode_rewards'],
                episode_lengths=self.training_history['episode_lengths'],
                exploration_rates=self.training_history['exploration_rates'],
                losses=self.training_history['losses'])
    
    def load_training_history(self, filename='training_history.npz'):
        """
        加载训练历史
        """
        data = np.load(filename)
        self.training_history = {
            'episode_rewards': data['episode_rewards'].tolist(),
            'episode_lengths': data['episode_lengths'].tolist(),
            'exploration_rates': data['exploration_rates'].tolist(),
            'losses': data['losses'].tolist()
        }
        
    def save_checkpoint(self, episode):
        """
        保存檢查點
        
        Args:
            episode: 當前訓練輪數
        """
        # 用零填充確保文件名排序正確
        ep_str = str(episode).zfill(6)
        
        # 保存模型
        model_path = os.path.join(MODEL_DIR, f'frontier_model_ep{ep_str}.h5')
        self.model.save(model_path)
        
        # 保存訓練歷史
        history_path = os.path.join(MODEL_DIR, f'training_history_ep{ep_str}.json')
        history_to_save = {
            'episode_rewards': [float(x) for x in self.training_history['episode_rewards']],
            'episode_lengths': [int(x) for x in self.training_history['episode_lengths']],
            'exploration_rates': [float(x) for x in self.training_history['exploration_rates']],
            'losses': [float(x) if x is not None else 0.0 for x in self.training_history['losses']]
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_to_save, f, indent=4)
        
        print(f"Saved checkpoint at episode {episode}")