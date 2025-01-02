import tensorflow as tf
import numpy as np

class MultiRobotNetworkModel:
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        """初始化多机器人网络模型
        
        Args:
            input_shape: 输入地图的形状，默认(84, 84, 1)
            max_frontiers: 最大frontier点数量，默认50
        """
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.model = self._build_model()
        self.target_model = self._build_model()
        
    def _build_cnn_block(self, inputs, filters, kernel_size):
        """构建CNN块,包含BN和残差连接"""
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # 添加残差连接(如果维度相同)
        if inputs.shape[-1] == filters:
            x = tf.keras.layers.Add()([inputs, x])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    def _build_model(self):
        """构建改进的网络模型"""
        # 1. 地图输入处理
        map_input = tf.keras.layers.Input(shape=self.input_shape, name='map_input')
        
        # 多尺度特征提取
        # Path 1: 细节特征
        x1 = self._build_cnn_block(map_input, 32, 3)
        x1 = tf.keras.layers.MaxPooling2D(2)(x1)
        
        # Path 2: 中等尺度特征
        x2 = self._build_cnn_block(map_input, 32, 5)
        x2 = tf.keras.layers.MaxPooling2D(2)(x2)
        
        # Path 3: 大尺度特征
        x3 = self._build_cnn_block(map_input, 32, 7)
        x3 = tf.keras.layers.MaxPooling2D(2)(x3)
        
        # 合并多尺度特征
        x = tf.keras.layers.Concatenate()([x1, x2, x3])
        
        # 继续处理合并的特征
        x = self._build_cnn_block(x, 64, 3)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = self._build_cnn_block(x, 64, 3)
        map_features = tf.keras.layers.Flatten()(x)
        
        # 2. Frontier输入处理
        frontier_input = tf.keras.layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        
        # 3. 机器人位置输入
        robot1_pos_input = tf.keras.layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos_input = tf.keras.layers.Input(shape=(2,), name='robot2_pos_input')
        
        # 计算机器人间的相对位置
        relative_pos = tf.keras.layers.Subtract()([robot1_pos_input, robot2_pos_input])
        
        # 4. Frontier注意力机制
        frontier_features = tf.keras.layers.Dense(64, activation='relu')(frontier_input)
        
        # 计算每个frontier相对于两个机器人的位置
        robot1_pos_expanded = tf.keras.layers.RepeatVector(self.max_frontiers)(robot1_pos_input)
        robot2_pos_expanded = tf.keras.layers.RepeatVector(self.max_frontiers)(robot2_pos_input)
        
        # 相对位置特征
        rel_to_robot1 = tf.keras.layers.Subtract()([frontier_input, robot1_pos_expanded])
        rel_to_robot2 = tf.keras.layers.Subtract()([frontier_input, robot2_pos_expanded])
        
        # 计算注意力权重
        attention_features = tf.keras.layers.Concatenate()([
            frontier_features,
            rel_to_robot1,
            rel_to_robot2
        ])
        
        attention_dense = tf.keras.layers.Dense(64, activation='relu')(attention_features)
        attention_scores = tf.keras.layers.Dense(1)(attention_dense)
        attention_weights = tf.keras.layers.Softmax(axis=1)(attention_scores)
        
        # 加权frontier特征
        weighted_frontiers = tf.keras.layers.Multiply()([frontier_features, attention_weights])
        frontier_context = tf.keras.layers.GlobalAveragePooling1D()(weighted_frontiers)
        
        # 5. 特征融合
        robot1_features = tf.keras.layers.Dense(64, activation='relu')(robot1_pos_input)
        robot2_features = tf.keras.layers.Dense(64, activation='relu')(robot2_pos_input)
        relative_features = tf.keras.layers.Dense(64, activation='relu')(relative_pos)
        
        # 合并所有特征
        combined = tf.keras.layers.Concatenate()([
            map_features,
            frontier_context,
            robot1_features,
            robot2_features,
            relative_features
        ])
        
        # 6. 深度特征处理
        shared = tf.keras.layers.Dense(512, activation='relu')(combined)
        shared = tf.keras.layers.Dropout(0.2)(shared)
        shared = tf.keras.layers.Dense(256, activation='relu')(shared)
        shared = tf.keras.layers.Dropout(0.2)(shared)
        
        # 7. 双重输出流
        # Robot 1输出
        robot1_stream = tf.keras.layers.Dense(256, activation='relu')(shared)
        robot1_value = tf.keras.layers.Dense(1)(robot1_stream)
        robot1_advantage = tf.keras.layers.Dense(self.max_frontiers)(robot1_stream)
        robot1_output = tf.keras.layers.Add()([
            robot1_value,
            tf.keras.layers.Subtract()([
                robot1_advantage,
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.backend.mean(x, axis=1, keepdims=True)
                )(robot1_advantage)
            ])
        ])
        
        # Robot 2输出
        robot2_stream = tf.keras.layers.Dense(256, activation='relu')(shared)
        robot2_value = tf.keras.layers.Dense(1)(robot2_stream)
        robot2_advantage = tf.keras.layers.Dense(self.max_frontiers)(robot2_stream)
        robot2_output = tf.keras.layers.Add()([
            robot2_value,
            tf.keras.layers.Subtract()([
                robot2_advantage,
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.backend.mean(x, axis=1, keepdims=True)
                )(robot2_advantage)
            ])
        ])
        
        # 8. 构建最终模型
        model = tf.keras.Model(
            inputs={
                'map_input': map_input,
                'frontier_input': frontier_input,
                'robot1_pos_input': robot1_pos_input,
                'robot2_pos_input': robot2_pos_input
            },
            outputs={
                'robot1': robot1_output,
                'robot2': robot2_output
            }
        )
        
        # 9. 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss={
                'robot1': tf.keras.losses.Huber(),
                'robot2': tf.keras.losses.Huber()
            }
        )
        
        return model
    
    def update_target_model(self):
        """更新目标网络的权重"""
        self.target_model.set_weights(self.model.get_weights())
    
    def predict(self, state, frontiers, robot1_pos, robot2_pos):
        """预测动作值
        
        Args:
            state: 地图状态
            frontiers: frontier点列表
            robot1_pos: 机器人1的位置
            robot2_pos: 机器人2的位置
            
        Returns:
            包含两个机器人动作值的字典
        """
        # 确保输入形状正确
        if len(state.shape) == 3:
            state = np.expand_dims(state, 0)
        if len(frontiers.shape) == 2:
            frontiers = np.expand_dims(frontiers, 0)
        if len(robot1_pos.shape) == 1:
            robot1_pos = np.expand_dims(robot1_pos, 0)
        if len(robot2_pos.shape) == 1:
            robot2_pos = np.expand_dims(robot2_pos, 0)
            
        return self.model.predict({
            'map_input': state,
            'frontier_input': frontiers,
            'robot1_pos_input': robot1_pos,
            'robot2_pos_input': robot2_pos
        })
    
    def train_on_batch(self, states, frontiers, robot1_pos, robot2_pos, 
                      robot1_targets, robot2_targets):
        """训练一个批次
        
        Args:
            states: 批次的地图状态
            frontiers: 批次的frontier点
            robot1_pos: 批次的机器人1位置
            robot2_pos: 批次的机器人2位置
            robot1_targets: 机器人1的目标Q值
            robot2_targets: 机器人2的目标Q值
            
        Returns:
            训练损失
        """
        return self.model.train_on_batch(
            {
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos
            },
            {
                'robot1': robot1_targets,
                'robot2': robot2_targets
            }
        )
    
    def save(self, path):
        """保存模型
        
        Args:
            path: 保存路径
        """
        self.model.save(path)
    
    def load(self, path):
        """加载模型
        
        Args:
            path: 模型路径
        """
        self.model = tf.keras.models.load_model(path)
        self.target_model = tf.keras.models.load_model(path)