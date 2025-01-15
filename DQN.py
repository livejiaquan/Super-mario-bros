import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class ReplayMemory:                                                              # 存儲和取樣訓練數據
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)                                     # 使用 deque 儲存數據，設置 maxlen，確保記憶體達到容量上限時，會自動移除最舊的經驗
                                                                                 # self.memory 是一個 deque，存儲了許多 tuple，每個 tuple 表示一條經驗，格式為(s,a,r,n_s,d)
    def push(self, state, action, reward, next_state, done):                     # 將經驗 (state, action, reward, next_state, done) 添加到記憶體
        self.memory.append((state, action, reward, next_state, done))    

    def sample(self, batch_size): 
        batch = random.sample(self.memory, batch_size)                           # batch = 從記憶體中隨機取樣的 batch_size 筆資料
        states, actions, rewards, next_states, dones = zip(*batch)               # zip(*batch) 會將多筆經驗中的同類型資料（如狀態、動作）組合在一起
                                                                                 # ex.batch取樣32條(s,a,r,n_s,d)，則 zip(*batch) 中 states=[s1,s2,...,s32],actions=[a1,a2,...,a32]
        return np.stack(states), actions, rewards, np.stack(next_states), dones  # np.stack 將 states 與 next_states 轉為 NumPy 陣列，方便後續運算

    def __len__(self):                                                           # 記憶體中目前儲存的經驗數量
        return len(self.memory)
    # 優先經驗回放


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]  # 取樣
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()

        # 解壓樣本為分別的批次數據
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones)), indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):  # 記憶體中目前儲存的經驗數量
        return len(self.memory)
class DQN:
    def __init__(self,
                 model,
                 state_dim, action_dim, 
                 learning_rate, gamma,
                 epsilon, target_update, device):
        self.device = device
        self.action_dim = action_dim
        
        self.gamma = gamma  
        self.epsilon = epsilon
        self.target_update = target_update
        self.update_count = 0

        # Initialize [Q-net] and target [Q-net]
        self.model = model
        self.q_net = self._build_net(state_dim, action_dim)                      #  [Q-net]，實際訓練的網路 (即時更新)
        self.tgt_q_net = self._build_net(state_dim, action_dim)                  # target [Q-net]，用於穩定訓練（延遲更新）
        self.tgt_q_net.load_state_dict(self.q_net.state_dict())                  # 複製 [Q-net] 的權重到 target [Q-net]

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
    
    # 定義神經網路構造函數
    def _build_net(self, state_dim, action_dim):                                 # 根據傳入的模型架構 (self.model) 初始化後的神經網路，並將其移到 device 上
        return self.model(state_dim,action_dim).to(self.device)
    
    # action 選擇函數
    def take_action(self, state,preferred_action=4, preferred_weight=10):
        # Exploration Unknown Policy(探索)
        if np.random.rand() < self.epsilon:
            if preferred_action is not None:
                # 建立加權隨機分佈
                weights = np.ones(self.action_dim)  # 初始化每個動作的權重為 1
                weights[preferred_action] = preferred_weight  # 增加特定動作的權重

                # 根據權重生成加權分佈的隨機動作
                probabilities = weights / weights.sum()  # 正規化為概率分佈
                return np.random.choice(self.action_dim, p=probabilities)
            else:
                # 如果沒有設置偏好動作，執行均勻隨機
                return np.random.randint(self.action_dim)
        # Exploitation Known Policy(利用)                                        # 隨機浮點數大於 epsilon 執行根據推理的動作（利用）
        state_x = torch.tensor([state], dtype=torch.float32, device=self.device)  # 單一 state 轉換為 PyTorch 張量
        with torch.no_grad():
            q_values = self.q_net(state_x)  # 使用 [Q-net] 計算該 state 下每個動作的 Q 值
            action_probs = F.softmax(q_values, dim=1)  # 通過 softmax 將 Q 值轉換為機率分佈

            # 確保 action_probs 是有效的機率分佈（總和為 1）
            action_probs = action_probs / action_probs.sum()

            # 將調整後的機率轉換為類別分佈
            action_dist = torch.distributions.Categorical(action_probs)
            return action_dist.sample().item()  # 從機率分佈中抽樣一個動作                                 # 從類別分佈中抽樣一個動作，並返回對應的索引（即選擇的動作）
        
    #　損失函數計算
    def get_loss(self, states, actions, rewards, next_states, dones):
        # Get current Q-values
        actions = actions.unsqueeze(1) 
        q_val = self.q_net(states).gather(1, actions).squeeze(1)                 # 計算當下的 Q-value
                                                                 
        # Get maximum expected Q-values
        next_q_val = self.tgt_q_net(next_states).max(dim=1)[0]                   # 計算 target Q-value 的最大值 
                                                               
        # Compute target Q-values [custom-reward]
        q_target = rewards + self.gamma * next_q_val * (1 - dones.float())       # 計算 target Q-value
        
        return torch.nn.functional.mse_loss(q_val, q_target.detach())                # 用均方誤差 (MSE) 計算 loss 

    def train_per_step(self, state_dict):
        # Convert one trajectory(s,a,r,n_s) to tensor
        states,actions,rewards,next_states,dones = self._state_2_tensor(state_dict)  # 將原本存儲於 Python 資料結構中的數據轉換為 PyTorch 張量

        # Compute loss 
        loss = self.get_loss(states, actions, rewards, next_states, dones)
        self.optimizer.zero_grad()                                               # 每次進行梯度更新之前，清除累積的梯度值
        loss.backward()                                                          # 利用計算的損失值進行反向傳播，計算每個參數的梯度
        self.optimizer.step()                                                    # 利用計算的梯度來更新 [Q-net] 的參數

        if self.update_count % self.target_update == 0:                          # runs.py 內定義 target_update=TARGET_UPDATE=50(更新頻率)
            self.tgt_q_net.load_state_dict(self.q_net.state_dict())              # 定期將 [Q-net] 的參數複製到 target [Q-net]

        self.update_count += 1
    
    def _state_2_tensor(self,state_dict):                                        # 將一條經驗軌跡 (s,a,r,n_s,d) 中的數據轉換為 PyTorch 張量
        states      = torch.tensor(state_dict['states'], dtype=torch.float32, device=self.device)
        actions     = torch.tensor(state_dict['actions'], dtype=torch.long, device=self.device)
        rewards     = torch.tensor(state_dict['rewards'], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(state_dict['next_states'], dtype=torch.float32, device=self.device)
        dones       = torch.tensor(state_dict['dones'], dtype=torch.float32, device=self.device)

        return states,actions,rewards,next_states,dones
    

class ACDQN:
    def __init__(self, model, state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.update_count = 0

        # Initialize Actor-Critic networks
        self.q_net = model(state_dim, action_dim).to(self.device)
        self.tgt_q_net = model(state_dim, action_dim).to(self.device)
        self.tgt_q_net.load_state_dict(self.q_net.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def take_action(self, state, preferred_action = None, preferred_weight = 10):
        # Exploration vs Exploitation
        if np.random.rand() < self.epsilon:
            if preferred_action is not None:
                # 建立加權隨機分佈
                weights = np.ones(self.action_dim)  # 初始化每個動作的權重為 1
                weights[preferred_action] = preferred_weight  # 增加特定動作的權重

                # 根據權重生成加權分佈的隨機動作
                probabilities = weights / weights.sum()  # 正規化為概率分佈
                return np.random.choice(self.action_dim, p=probabilities)
            else:
                # 如果沒有設置偏好動作，執行均勻隨機
                return np.random.randint(self.action_dim)
        state = np.expand_dims(state, axis=0)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_values, _ = self.q_net(state)
        return torch.argmax(action_values, dim=1).item()

    def get_loss(self, states, actions, rewards, next_states, dones):
        # Current Q-values
        actions = actions.unsqueeze(1)
        q_values, state_values = self.q_net(states)  # Actor-Critic outputs
        q_values = q_values.gather(1, actions).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values, next_state_values = self.tgt_q_net(next_states)
            next_q_values = next_q_values.max(dim=1)[0]  # Max over actions

        # Critic (state value) to stabilize training
        q_target = rewards + self.gamma * (1 - dones.float()) * next_state_values.squeeze()

        # Loss
        q_loss = F.mse_loss(q_values, q_target)
        critic_loss = F.mse_loss(state_values.squeeze(), q_target)

        return q_loss + critic_loss

    def train_per_step(self, state_dict):
        # Prepare batch
        states, actions, rewards, next_states, dones = self._state_2_tensor(state_dict)

        # Compute loss and update network
        loss = self.get_loss(states, actions, rewards, next_states, dones)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        if self.update_count % self.target_update == 0:
            self.tgt_q_net.load_state_dict(self.q_net.state_dict())
        self.update_count += 1

    def _state_2_tensor(self, state_dict):
        states = torch.tensor(state_dict['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(state_dict['actions'], dtype=torch.long, device=self.device)
        rewards = torch.tensor(state_dict['rewards'], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(state_dict['next_states'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(state_dict['dones'], dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, dones