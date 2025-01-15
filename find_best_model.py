import numpy as np
import torch
import os

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from movement import CUSTOM_MOVEMENT
from utils import preprocess_frame
from model_AC import ActorCriticCNN
from DQN import ACDQN
from skipframe import SkipFrame

# ========== Config ===========
MODEL_DIR = "ckpt_test/16/"  # 模型存放的資料夾路徑
skipframe_values = [2, 4, 8]  # 不同的 skipframe 值
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')  # 建立《超級瑪利歐兄弟》的遊戲環境(第1個世界的第1關)

# SIMPLE_MOVEMENT可以自行定義
env = JoypadSpace(env, CUSTOM_MOVEMENT)

device = torch.device("mps")  # 設定運算設備為MPS
OBS_SHAPE = (1, 84, 84)  # 遊戲畫面轉換為 (1, 84, 84) 的灰階圖像
N_ACTIONS = len(CUSTOM_MOVEMENT)
VISUALIZE = True  # 是否在每回合中顯示遊戲畫面

best_total_reward = -float('inf')  # 初始化最佳總獎勳為負無窮
best_model_file = None  # 儲存最佳模型檔案
best_skipframe = None  # 儲存最佳skipframe值

# 迴圈測試不同的skipframe值
for skipframe in skipframe_values:
    env_with_skip = SkipFrame(env, skip=skipframe)

    # ========== Initialize DQN ===========
    dqn = ACDQN(
        model=ActorCriticCNN,
        state_dim=OBS_SHAPE,
        action_dim=N_ACTIONS,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=0.1,  # 設為 0.0 表示完全利用當下的策略
        target_update=100,  # target [Q-net] 更新的頻率
        device=device
    )

    # 讀取資料夾中的所有模型檔案
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]

    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)

        # 載入模型權重
        if os.path.exists(model_path):
            try:
                model_weights = torch.load(model_path, map_location=device)  # 載入模型權重
                dqn.q_net.load_state_dict(model_weights)  # 應用權重到模型
                dqn.q_net.eval()  # 設定模型為評估模式
            except Exception as e:
                print(f"Failed to load model weights: {e}")
                raise
        else:
            print(f"Model file not found: {model_path}")
            continue

        # ========== Evaluation Loop ===========
        total_rewards = 0
        state = env_with_skip.reset()  # 重置環境
        state = preprocess_frame(state)
        state = np.expand_dims(state, axis=0)  # 新增 channel dimension
        state = np.expand_dims(state, axis=0)  # 新增 batch dimension
        done = False
        total_reward = 0

        while not done:
            # 選擇行動
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_logits, _ = dqn.q_net(state_tensor)  # 計算動作 logits
                action_probs = torch.softmax(action_logits, dim=1)
                action = torch.argmax(action_probs, dim=1).item()

            next_state, reward, done, info = env_with_skip.step(action)  # 環境進行一步
            next_state = preprocess_frame(next_state)
            next_state = np.expand_dims(next_state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)

            total_reward += reward
            state = next_state

            if VISUALIZE:
                env_with_skip.render()

        # 如果當前的total reward高於最高紀錄，則更新最佳組合
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_model_file = model_file
            best_skipframe = skipframe
        print(f"Path:{model_path} - Skipframes: {skipframe} Total reward: {total_reward}")

# 印出最佳組合
print(f"\nBest model: {best_model_file} with skipframe={best_skipframe} - Total Reward: {best_total_reward}")

env.close()