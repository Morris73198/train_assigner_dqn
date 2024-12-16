import os
import numpy as np
import numpy.ma as ma
from scipy import spatial
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from ..utils.inverse_sensor_model import inverse_sensor_model
from ..utils.astar import astar
import random
import heapq
from ..config import ROBOT_CONFIG, REWARD_CONFIG

class Robot:
    def __init__(self, index_map, train, plot):
        """初始化机器人环境"""
        self.mode = train
        self.plot = plot
        
        # 设置路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        
        if self.mode:
            self.map_dir = os.path.join(base_dir, 'robot_rl/data', 'DungeonMaps', 'train')
        else:
            self.map_dir = os.path.join(base_dir, 'robot_rl/data', 'DungeonMaps', 'test')
            
        # 确保目录存在
        os.makedirs(self.map_dir, exist_ok=True)
        
        self.map_list = os.listdir(self.map_dir)
        if not self.map_list:
            raise FileNotFoundError(f"No map files found in {self.map_dir}")
            
        self.map_number = np.size(self.map_list)
        if self.mode:
            random.shuffle(self.map_list)
            
        self.li_map = index_map
        
        # 初始化地图和机器人位置
        self.global_map, self.robot_position = self.map_setup(
            os.path.join(self.map_dir, self.map_list[self.li_map]))
        self.robot_position = self.robot_position.astype(np.int64)

        self.op_map = np.ones(self.global_map.shape) * 127
        self.map_size = np.shape(self.global_map)
        
        # 从配置文件加载参数
        self.movement_step = ROBOT_CONFIG['movement_step']
        self.finish_percent = ROBOT_CONFIG['finish_percent']
        self.sensor_range = ROBOT_CONFIG['sensor_range']
        self.robot_size = ROBOT_CONFIG['robot_size']
        self.local_size = ROBOT_CONFIG['local_size']
        
        # 状态记录
        self.old_position = np.zeros([2])
        self.old_op_map = np.empty([0])
        self.current_target_frontier = None
        self.is_moving_to_target = False  # 新增:標記是否正在移動到目標
        self.steps = 0
        
        # 地图点和自由空间
        self.t = self.map_points(self.global_map)
        self.free_tree = spatial.KDTree(self.free_points(self.global_map).tolist())
        
        # 可视化相关
        if self.plot:
            self.xPoint = np.array([self.robot_position[0]])
            self.yPoint = np.array([self.robot_position[1]])
            self.x2frontier = np.empty([0])
            self.y2frontier = np.empty([0])

    def begin(self):
        """初始化并返回初始状态"""
        self.op_map = self.inverse_sensor(
            self.robot_position, self.sensor_range, self.op_map, self.global_map)
            
        step_map = self.robot_model(
            self.robot_position, self.robot_size, self.t, self.op_map)
            
        map_local = self.local_map(
            self.robot_position, step_map, self.map_size, self.sensor_range + self.local_size)
        
        resized_map = resize(map_local, (84, 84))
        state = np.expand_dims(resized_map, axis=-1)
        
        if self.plot:
            self.plot_env()
            
        return state

    def move_to_frontier(self, target_frontier):
        """移動到指定的frontier點"""
        # 如果正在執行移動且目標點沒有改變,繼續使用當前目標
        if self.is_moving_to_target and np.array_equal(self.current_target_frontier, target_frontier):
            target_frontier = self.current_target_frontier
        else:
            # 新的移動任務
            self.current_target_frontier = target_frontier.copy()
            self.is_moving_to_target = True
        
        # 獲取路徑
        path = self.astar_path(
            self.op_map, 
            self.robot_position.astype(np.int32),
            target_frontier.astype(np.int32),
            safety_distance=ROBOT_CONFIG['safety_distance']
        )
        
        if path is None:
            self.is_moving_to_target = False  # 重置移動狀態
            return self.get_observation(), -1, True
            
        path = self.simplify_path(path, ROBOT_CONFIG['path_simplification'])
        
        total_reward = 0
        done = False
        next_state = None
        
        path_points = path.T
        step_size = ROBOT_CONFIG['max_frontier_skip']
        
        for i in range(0, len(path_points), step_size):
            target_point = path_points[min(i + step_size - 1, len(path_points) - 1)]
            move_vector = target_point - self.robot_position
            
            dist = np.linalg.norm(move_vector)
            if dist > ROBOT_CONFIG['movement_step']:
                move_vector = move_vector * (ROBOT_CONFIG['movement_step'] / dist)
                
            next_state, reward, step_done = self.execute_movement(move_vector)
            total_reward += reward
            
            if step_done:
                done = True
                break
                
            # 檢查是否到達目標點附近
            dist_to_target = np.linalg.norm(self.robot_position - target_frontier)
            if dist_to_target < ROBOT_CONFIG['movement_step']:
                done = True
                break
        
        # 如果完成移動或遇到障礙,重置移動狀態
        if done:
            self.is_moving_to_target = False
            self.current_target_frontier = None
            
        return next_state, total_reward, done

    def execute_movement(self, move_vector):
        """执行移动"""
        old_position = self.robot_position.copy()
        old_op_map = self.op_map.copy()
        
        # 更新位置
        new_position = self.robot_position + move_vector
        self.robot_position = np.round(new_position).astype(np.int64)
        
        # 边界检查
        self.robot_position[0] = np.clip(self.robot_position[0], 0, self.map_size[1]-1)
        self.robot_position[1] = np.clip(self.robot_position[1], 0, self.map_size[0]-1)
        
        # 碰撞检测
        collision_points, collision_index = self.fast_collision_check(
            old_position, self.robot_position, self.map_size, self.global_map)
        
        if collision_index:
            self.robot_position = self.nearest_free(self.free_tree, collision_points)
            reward = REWARD_CONFIG['collision_penalty']
            done = True
        else:
            self.op_map = self.inverse_sensor(
                self.robot_position, self.sensor_range, 
                self.op_map, self.global_map
            )
            reward = self.calculate_fast_reward(old_op_map, self.op_map, move_vector)
            done = False
        
        self.steps += 1
        if self.plot and self.steps % ROBOT_CONFIG['plot_interval'] == 0:
            self.xPoint = np.append(self.xPoint, self.robot_position[0])
            self.yPoint = np.append(self.yPoint, self.robot_position[1])
            self.plot_env()
        
        return self.get_observation(), reward, done

    def calculate_fast_reward(self, old_op_map, new_op_map, move_vector):
        """计算奖励"""
        # 探索奖励
        explored_diff = float(
            np.sum(new_op_map == 255) - np.sum(old_op_map == 255)
        ) / 14000 * REWARD_CONFIG['exploration_weight']
        
        # 移动惩罚
        movement_cost = REWARD_CONFIG['movement_penalty'] * np.linalg.norm(move_vector)
        
        # 目标导向奖励
        if self.current_target_frontier is not None:
            distance_to_target = np.linalg.norm(
                self.current_target_frontier - self.robot_position)
            progress_reward = -0.01 * distance_to_target
        else:
            progress_reward = 0
        
        total_reward = explored_diff + movement_cost + progress_reward
        return np.clip(total_reward, -1, 1)

    def map_setup(self, location):
        """设置地图和机器人初始位置"""
        global_map = (io.imread(location, 1) * 255).astype(int)
        robot_location = np.nonzero(global_map == 208)
        robot_location = np.array([np.array(robot_location)[1, 0], 
                                 np.array(robot_location)[0, 0]])
        global_map = (global_map > 150)
        global_map = global_map * 254 + 1
        return global_map, robot_location

    def map_points(self, map_glo):
        """生成地图点"""
        map_x = map_glo.shape[1]
        map_y = map_glo.shape[0]
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def local_map(self, robot_location, map_glo, map_size, local_size):
        """获取局部地图"""
        minX = int(robot_location[0] - local_size)
        maxX = int(robot_location[0] + local_size)
        minY = int(robot_location[1] - local_size)
        maxY = int(robot_location[1] + local_size)

        minX = max(0, minX)
        maxX = min(map_size[1], maxX)
        minY = max(0, minY)
        maxY = min(map_size[0], maxY)

        return map_glo[minY:maxY, minX:maxX]

    def free_points(self, op_map):
        """获取自由空间点"""
        index = np.where(op_map == 255)
        return np.asarray([index[1], index[0]]).T

    def nearest_free(self, tree, point):
        """找到最近的自由点"""
        pts = np.atleast_2d(point)
        index = tuple(tree.query(pts)[1])
        return tree.data[index]

    def robot_model(self, position, robot_size, points, map_glo):
        """机器人模型"""
        map_copy = map_glo.copy()
        robot_points = self.range_search(position, robot_size, points)
        for point in robot_points:
            y, x = point[::-1].astype(int)
            if 0 <= y < map_copy.shape[0] and 0 <= x < map_copy.shape[1]:
                map_copy[y, x] = 76
        return map_copy

    def range_search(self, position, r, points):
        """范围搜索"""
        diff = points - position
        dist_sq = np.sum(diff * diff, axis=1)
        return points[dist_sq <= r * r]

    def fast_collision_check(self, start_point, end_point, map_size, map_glo):
        """简化的碰撞检测"""
        start = np.round(start_point).astype(int)
        end = np.round(end_point).astype(int)
        
        # 检查终点是否有效
        if not (0 <= end[0] < map_size[1] and 0 <= end[1] < map_size[0]):
            return np.array([end]).reshape(1, 2), True
            
        if map_glo[end[1], end[0]] == 1:
            return np.array([end]).reshape(1, 2), True
        
        # 简化的路径检查
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        steps = max(abs(dx), abs(dy))
        
        if steps == 0:
            return np.array([[-1, -1]]).reshape(1, 2), False
            
        x_step = dx / steps
        y_step = dy / steps
        
        # 只检查几个关键点
        check_points = np.linspace(0, steps, min(5, steps + 1))
        for t in check_points:
            x = int(start[0] + x_step * t)
            y = int(start[1] + y_step * t)
            
            if not (0 <= x < map_size[1] and 0 <= y < map_size[0]):
                return np.array([[x, y]]).reshape(1, 2), True
                
            if map_glo[y, x] == 1:
                return np.array([[x, y]]).reshape(1, 2), True
        
        return np.array([[-1, -1]]).reshape(1, 2), False

    def inverse_sensor(self, robot_position, sensor_range, op_map, map_glo):
        """逆向传感器模型"""
        return inverse_sensor_model(
            int(robot_position[0]), int(robot_position[1]), 
            sensor_range, op_map, map_glo)

    def frontier(self, op_map, map_size, points):
        """获取frontier点"""
        y_len, x_len = map_size
        mapping = (op_map == 127).astype(int)
        
        # 添加边界填充
        mapping = np.pad(mapping, ((1,1), (1,1)), 'constant')
        
        # 计算邻域和
        fro_map = (
            mapping[2:, 1:x_len+1] +    # 下
            mapping[:y_len, 1:x_len+1] + # 上
            mapping[1:y_len+1, 2:] +     # 右
            mapping[1:y_len+1, :x_len] + # 左
            mapping[:y_len, 2:] +        # 右上
            mapping[2:, :x_len] +        # 左下
            mapping[2:, 2:] +            # 右下
            mapping[:y_len, :x_len]      # 左上
        )
        
        # 找到满足条件的点
        free_space = op_map.ravel(order='F') == 255
        frontier_condition = (1 < fro_map.ravel(order='F')) & (fro_map.ravel(order='F') < 8)
        valid_points = points[np.where(free_space & frontier_condition)[0]]
        
        # 优化：只返回间隔足够大的frontier点
        if len(valid_points) > 0:
            selected_points = [valid_points[0]]
            min_dist = ROBOT_CONFIG['min_frontier_dist']
            
            for point in valid_points[1:]:
                # 检查与已选点的距离
                distances = [np.linalg.norm(point - p) for p in selected_points]
                if min(distances) > min_dist:
                    selected_points.append(point)
            
            return np.array(selected_points).astype(int)
        
        return valid_points.astype(int)

    def get_frontiers(self):
        """獲取所有可用的frontiers點"""
        # 如果正在移動到目標,只返回當前目標
        if self.is_moving_to_target and self.current_target_frontier is not None:
            return np.array([self.current_target_frontier])
            
        frontiers = self.frontier(self.op_map, self.map_size, self.t)
        if len(frontiers) == 0:
            return np.zeros((0, 2))
            
        # 按照到機器人的距離排序
        distances = np.linalg.norm(frontiers - self.robot_position, axis=1)
        sorted_indices = np.argsort(distances)
        return frontiers[sorted_indices]

    def plot_env(self):
        """绘制环境状态可视化"""
        plt.cla()
        plt.imshow(self.op_map, cmap='gray')
        plt.axis((0, self.map_size[1], self.map_size[0], 0))
        
        # 绘制机器人路径
        plt.plot(self.xPoint, self.yPoint, 'b-', linewidth=2, label='Robot Path')
        
        # 绘制所有frontiers
        frontiers = self.get_frontiers()
        if len(frontiers) > 0:
            plt.scatter(frontiers[:, 0], frontiers[:, 1], 
                    c='red', marker='*', s=100, label='Frontiers')
        
        # 绘制当前目标frontier和路径
        if self.current_target_frontier is not None:
            plt.plot(self.current_target_frontier[0], self.current_target_frontier[1], 
                    'go', markersize=10, label='Target Frontier')
            
            path = self.astar_path(
                self.op_map,
                self.robot_position.astype(np.int32),
                self.current_target_frontier.astype(np.int32)
            )
            
            if path is not None and path.shape[1] > 1:
                path_x = path[0, :]
                path_y = path[1, :]
                plt.plot(path_x, path_y, 'g--', linewidth=2, 
                        alpha=0.8, label='Planned Path')
        
        # 绘制机器人当前位置和起始位置
        plt.plot(self.robot_position[0], self.robot_position[1], 
                'mo', markersize=8, label='Current Position')
        plt.plot(self.xPoint[0], self.yPoint[0], 
                'co', markersize=8, label='Start Position')
        
        # 添加图例和探索进度
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        explored_ratio = np.sum(self.op_map == 255) / np.sum(self.global_map == 255)
        plt.title(f'Exploration Progress: {explored_ratio:.1%}')
        
        plt.pause(0.01)

    def astar(self, op_map, start, goal):
        """A*路径规划"""
        start = tuple(start)
        goal = tuple(goal)
        
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                    (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))
        
        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 边界检查
                if not (0 <= neighbor[0] < self.map_size[1] and 
                       0 <= neighbor[1] < self.map_size[0]):
                    continue
                    
                # 障碍物检查
                if op_map[neighbor[1]][neighbor[0]] == 1:
                    continue
                
                # 移动代价计算，对角线移动代价更高
                move_cost = ROBOT_CONFIG['diagonal_weight'] if dx != 0 and dy != 0 else 1
                tentative_g_score = gscore[current] + move_cost
                
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue
                
                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        
        return None

    def heuristic(self, a, b):
        """启发式函数：使用对角线距离"""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        D = 1  # 直线移动代价
        D2 = ROBOT_CONFIG['diagonal_weight']  # 对角线移动代价
        return D * max(dx, dy) + (D2 - D) * min(dx, dy)

    def astar_path(self, op_map, start, goal, safety_distance=None):
        """获取A*路径"""
        if safety_distance is None:
            safety_distance = ROBOT_CONFIG['safety_distance']
            
        path = self.astar(op_map, start, goal)
        if path is None:
            return None
            
        path = np.array(path).T
        return self.simplify_path(path, ROBOT_CONFIG['path_simplification'])

    def simplify_path(self, path, threshold):
        """路径简化"""
        if path.shape[1] <= 2:
            return path
            
        def point_line_distance(point, start, end):
            if np.all(start == end):
                return np.linalg.norm(point - start)
                
            line_vec = end - start
            point_vec = point - start
            line_len = np.linalg.norm(line_vec)
            line_unit_vec = line_vec / line_len
            projection_length = np.dot(point_vec, line_unit_vec)
            
            if projection_length < 0:
                return np.linalg.norm(point - start)
            elif projection_length > line_len:
                return np.linalg.norm(point - end)
            else:
                return np.linalg.norm(point_vec - projection_length * line_unit_vec)
        
        def simplify_recursive(points, epsilon, mask):
            dmax = 0
            index = 0
            end = len(points) - 1
            
            for i in range(1, end):
                d = point_line_distance(points[i], points[0], points[end])
                if d > dmax:
                    index = i
                    dmax = d
            
            if dmax > epsilon:
                mask1 = mask.copy()
                mask2 = mask.copy()
                simplify_recursive(points[:index + 1], epsilon, mask1)
                simplify_recursive(points[index:], epsilon, mask2)
                
                for i in range(len(mask)):
                    mask[i] = mask1[i] if i <= index else mask2[i - index]
            else:
                for i in range(1, end):
                    mask[i] = False
        
        points = path.T
        mask = np.ones(len(points), dtype=bool)
        simplify_recursive(points, threshold, mask)
        
        return path[:, mask]

    def check_completion(self):
        """检查探索是否完成"""
        exploration_ratio = np.sum(self.op_map == 255) / np.sum(self.global_map == 255)
        
        if exploration_ratio > self.finish_percent:
            self.li_map += 1
            if self.li_map >= self.map_number:
                self.li_map = 0
                return True, True  # 完成所有地图
                
            self.__init__(self.li_map, self.mode, self.plot)
            return True, False  # 完成当前地图
            
        return False, False
    
    
    def reset(self):
        """重置环境到新地图"""
        self.li_map += 1
        if self.li_map >= self.map_number:
            self.li_map = 0
            
        # 初始化新地图
        self.__init__(self.li_map, self.mode, self.plot)
        return self.begin()

    def check_done(self):
        """检查是否需要结束当前回合"""
        # 检查探索进度
        exploration_ratio = np.sum(self.op_map == 255) / np.sum(self.global_map == 255)
        if exploration_ratio > self.finish_percent:
            return True
            
        # 检查是否还有可探索的frontiers
        frontiers = self.get_frontiers()
        if len(frontiers) == 0:
            return True
            
        return False

    def get_observation(self):
        """获取当前观察状态"""
        step_map = self.robot_model(
            self.robot_position, self.robot_size, self.t, self.op_map)
            
        map_local = self.local_map(
            self.robot_position, step_map, self.map_size, 
            self.sensor_range + self.local_size)
            
        resized_map = resize(map_local, (84, 84))
        return np.expand_dims(resized_map, axis=-1)

    def get_exploration_progress(self):
        """获取探索进度"""
        return np.sum(self.op_map == 255) / np.sum(self.global_map == 255)

    def get_state_info(self):
        """获取当前状态信息"""
        return {
            'position': self.robot_position.copy(),
            'map': self.op_map.copy(),
            'frontiers': self.get_frontiers(),
            'target_frontier': self.current_target_frontier,
            'exploration_progress': self.get_exploration_progress()
        }

    def set_state(self, state_info):
        """设置状态"""
        self.robot_position = state_info['position'].copy()
        self.op_map = state_info['map'].copy()
        self.current_target_frontier = state_info['target_frontier']
        
        if self.plot:
            self.plot_env()