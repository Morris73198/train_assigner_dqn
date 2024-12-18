import tensorflow as tf
import numpy as np

class FrontierNetworkModel:
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.model = self._build_model()
        self.target_model = self._build_model()
        
    def _build_model(self):
        # 輸入地圖
        map_input = tf.keras.layers.Input(shape=(84, 84, 1), name='map_input')
        
        # CNN處理地图
        x = tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu')(map_input)
        x = tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
        map_features = tf.keras.layers.Flatten()(x)
        
        # Frontier输入分支
        frontier_input = tf.keras.layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        
        # 處理frontier位置
        frontier_features = tf.keras.layers.Dense(128, activation='relu')(frontier_input)
        frontier_features = tf.keras.layers.Flatten()(frontier_features)
        
        # 特蒸融合
        combined = tf.keras.layers.Concatenate()([map_features, frontier_features])
        
        # 全连連接層
        x = tf.keras.layers.Dense(512, activation='relu')(combined)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        
        # 輸出層
        outputs = tf.keras.layers.Dense(50, activation='linear')(x)
        
        model = tf.keras.Model(inputs={'map_input': map_input, 
                                    'frontier_input': frontier_input}, 
                            outputs=outputs)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                    loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def predict(self, state, frontiers):
        return self.model.predict({
            'map_input': state,
            'frontier_input': frontiers
        })
        
    def train_on_batch(self, states, frontiers, targets):
        return self.model.train_on_batch([states, frontiers], targets)
        
    def save(self, path):
        self.model.save(path)
        
    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.target_model = tf.keras.models.load_model(path)