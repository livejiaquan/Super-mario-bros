import os
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import time

import gym_super_mario_bros                                      #導入gym_super_mario_bros，這是一個基於 Gym 的模組，用於模擬《Super Mario Bros》遊戲環境。
from nes_py.wrappers import JoypadSpace                          #從nes_py中導入JoypadSpace，用於限制遊戲中可用的按鈕動作（例如僅允許「移動右」或「跳躍」的動作集合）。
from movement import CUSTOM_MOVEMENT         #從 gym_super_mario_bros中導入SIMPLE_MOVEMENT，這是一個預定義的按鈕動作集合（如「右移」、「跳躍」等），用於控制 Mario 的行為。
from skipframe import SkipFrame                                                               #簡化動作空間 NES 控制器有 8 個按鍵（上下左右、A、B、Select、Start），可能的按鍵組合數非常大

from utils import preprocess_frame                               #用於對遊戲的畫面進行預處理，例如灰階化、調整大小等，將其轉換為適合神經網路輸入的格式
from reward import *                                             #模組中導入所有函式，這些函式用於設計和計算自定義獎勵（例如根據 Mario 的硬幣數量、水平位移等來計算獎勵）。
from model_AC import ActorCriticCNN                                      #自定義的卷積神經網路模型，用於處理遊戲畫面並生成動作決策
# from DQN import DQN, ReplayMemory                                #用於執行強化學習的主要邏輯 DQN模組中導入回放記憶體，用於存儲和抽取遊戲的狀態、動作、獎勵等樣本，提升訓練穩定性。
from DQN import ACDQN, PrioritizedReplayMemory

ROUND = 205
MODEL_LOAD_PATH = '/Users/jiaquan/Desktop/JQ_NCKU/1131_Computer Vision In Deep Learning Implementation And Its Applications/Final_Project/Super-mario-bros/ckpt_test/202/final_best_reward_31266_best_distance_3161.pth'

# ========== config ===========
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')   #
env = JoypadSpace(env, CUSTOM_MOVEMENT)
env = SkipFrame(env, skip = 8)

#========= basic train config==============================================
LR = 0.0003
BATCH_SIZE = 32                 #達到batch size更新主網路參數 達到50次更新目標網路的參數
GAMMA = 0.92                    #控制模型對長期獎勵和短期獎勵的權衡 gamma靠近1 模型更重視長期獎勵
MEMORY_SIZE = 100000             #用來儲存，遊戲過程中的記錄 如果存超過了 會刪除最早進來的
EPSILON_START = 0.3
EPSILON_DECAY = 0.9999
EPSILON_END = 0.2               #在訓練過程中，會逐漸從探索（隨機選擇動作）轉向利用（選擇模型預測的最佳動作）。
                                #EPSILON的值會隨著訓練進展逐漸下降，直到達到此最小值0.3
                                #即訓練後期仍保留 30% 的探索概率，避免模型陷入局部最優解
TARGET_UPDATE = 100              #每隔幾回合去更新目標網路的權重
TOTAL_TIMESTEPS = 20000          #總訓練的回合數
VISUALIZE = True                #是否在訓練過程中渲染遊戲畫面 顯示遊戲畫面
MAX_STAGNATION_STEPS = 1000       # Max steps without x_pos change 500
device = torch.device("cpu")

# ========================DQN Initialization==========================================
obs_shape = (1, 84, 84)                         #obs_shape = (1, 84, 84)
n_actions = len(CUSTOM_MOVEMENT)                #定義動作空間大小，使用SIMPLE_MOVEMENT中的動作數量（例如向右移動、跳躍等）
model = ActorCriticCNN                               #指定模型架構為CustomCNN用於處理圖像並預測各動作的 Q 值
dqn = ACDQN(                                      #初始化 DQN agent
    model=model,
    state_dim=obs_shape,                        #狀態空間大小
    action_dim=n_actions,                       #動作空間大小
    learning_rate=LR,                           #學習率
    gamma=GAMMA,                                #折扣因子，用於計算未來獎勵
    epsilon=EPSILON_START,                        #初始探索率
    target_update=TARGET_UPDATE,                #目標網路更新頻率
    device=device
)
if MODEL_LOAD_PATH:
    try:                                                                  # 檢查模型檔案是否存在：
        model_weights = torch.load(MODEL_LOAD_PATH, map_location=device)       #  若存在，嘗試載入模型權重
        dqn.q_net.load_state_dict(model_weights)                          #    載入成功，應用到模型
        dqn.epsilon=EPSILON_START
        print(f"Model loaded successfully from {MODEL_LOAD_PATH}")             #  若不存在，則FileNotFoundError
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        raise

memory = PrioritizedReplayMemory(MEMORY_SIZE)              #創建經驗回放記憶體，用於存儲狀態轉移
step = 0                                        #記錄總步數
best_reward = -float('inf')                     # 儲存最佳累積獎勵Track the best reward in each SAVE_INTERVAL  
best_distance = 0
cumulative_reward = 0                           # 當前時間步的總累積獎勵Track cumulative reward for the current timestep
reward_per_episode = []
distance_per_timestep = []


#=======================訓練開始============================
progress = tqdm(range(1, TOTAL_TIMESTEPS + 1), desc="Training Progress")
for timestep in progress:  #主訓練迴圈，進行TOTAL_TIMESTEPS次迭代
    state = env.reset()                                                         #重置遊戲環境，獲取初始狀態
    state = preprocess_frame(state)                                             #使用preprocess_frame 將畫面處理為灰階、縮放為84x84
    state = np.expand_dims(state, axis=0)                                       #新增一個維度，適配模型輸入

    done = False                                                                #表示當前遊戲是否結束
    prev_info = {                                                               #用於追蹤遊戲狀態（如水平位置、得分、硬幣數量）
        "x_pos": 0,  # Starting horizontal position (int).
        "y_pos": 0,  # Starting vertical position (int).
        "score": 0,  # Initial score is 0 (int).
        "coins": 0,  # Initial number of collected coins is 0 (int).
        "time": 400,  # Initial time in most levels of Super Mario Bros is 400 (int).
        "flag_get": False,  # Player has not yet reached the end flag (bool).
        "life": 3  # Default initial number of lives is 3 (int).
    }
    cumulative_custom_reward = 0                                                  #自定義獎勵總和
    cumulative_reward = 0 
    stagnation_time = 0                                                           #stagnation_time記錄遊戲角色在水平方向的停滯時間
    distance = 0
    custom_reward = 0.0
    #開始一個回合的遊戲循環
    while not done:
        action = dqn.take_action(state)                                           #輸入目前狀態，交給DQN去做下一步
        next_state, reward, done, info = env.step(action)                         #執行動作並從環境中獲取下一狀態、回報、遊戲結束標記、以及遊戲資訊 
       
        # preprocess image state 將下一狀態進行預處理並調整為適合模型的形狀
        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=0)

        cumulative_reward += reward   #更新累積獎勵

        # ===========================調用 reward.py 中的獎勵函數  包括硬幣獎勵、水平位移獎勵、擊敗敵人等
        custom_reward = calculate_coin_reward(info, reward, prev_info)
        custom_reward = calculate_vertical_movement_reward(info, custom_reward, prev_info)
        custom_reward = calculate_horizontal_movement_reward(info, custom_reward, prev_info)
        custom_reward, distance = calculate_speed_based_reward(info, custom_reward, distance)
        custom_reward = calculate_goal_completion_reward(info, custom_reward)
        custom_reward = calculate_score_reward(info, custom_reward, prev_info)
        custom_reward = calculate_altitude_bonus(info, custom_reward)
        custom_reward = apply_death_penalty(info, custom_reward, prev_info)
        custom_reward = calculate_survival_time_reward(info, custom_reward, prev_info)

        # ===========================Check for x_pos stagnation  如果角色的水平位置未改變超過MAX_STAGNATION_STEPS則強制結束本局遊戲
        if abs(info["x_pos"] - prev_info["x_pos"]) < 2:
            stagnation_time += 1
            if stagnation_time >= MAX_STAGNATION_STEPS:
                custom_reward = apply_stagnation_penalty(custom_reward)
                print(f"Timestep {timestep} - Early stop triggered due to x_pos stagnation.")
                done = True
        else:
            stagnation_time = 0
        # ==========================
        cumulative_custom_reward += custom_reward // 1
        progress.set_postfix({"stagnation_time":stagnation_time,"reward":cumulative_custom_reward})
        #===========================Store transition in memory 將狀態轉移 (state, action, reward, next_state, done) 存入記憶體
        memory.push(state, action, custom_reward //1, next_state, done)      #使用自訂義獎勵
        # memory.push(state, action, reward, next_state, done)                  #使用其預設好的獎勵
        #更新當前狀態
        state = next_state

        #==============================Train DQN 當記憶體中樣本數量達到批次大小時，從記憶體中隨機抽取一批樣本進行網路更新
        if len(memory) >= BATCH_SIZE:
            batch, indices, weights = memory.sample(BATCH_SIZE)
            state_dict = {
                'states': batch[0],  # states 批次
                'actions': batch[1],  # actions 批次
                'rewards': batch[2],  # rewards 批次
                'next_states': batch[3],  # next_states 批次
                'dones': batch[4],  # dones 批次
            }
            dqn.train_per_step(state_dict)
        
        #================================更新狀態訊息
        prev_info = info
        step += 1

        if VISUALIZE:                                   #渲染當前遊戲畫面
            env.render(mode='human')

    # Print cumulative reward for the current timestep
    print(f"Timestep {timestep} - Total Reward: {cumulative_reward} - Total Custom Reward: {cumulative_custom_reward} - EPSILON: {dqn.epsilon:.4f} - Distance: {distance}")
    # Update epsilon
    dqn.epsilon = max(EPSILON_END, dqn.epsilon * EPSILON_DECAY)     # 隨著時間逐漸減少探索率               
    #訓練前就設定:代理的探索能力會立即降低，可能在策略還不完善時過早專注於利用，會影響最終的學習效果
    #如果當前累積獎勵超過歷史最佳值，保存模型的權重 每次超過最佳值就會保留一次
    #要改成自定義獎勵
    reward_per_episode.append(cumulative_custom_reward)
    distance_per_timestep.append(distance)
    if cumulative_custom_reward > best_reward or distance > best_distance:
        if cumulative_custom_reward > best_reward:
            best_reward = cumulative_custom_reward
        elif distance > best_distance:
            best_distance = distance
        os.makedirs(f"ckpt_test/{ROUND}", exist_ok=True)
        #命名邏輯是採第幾步+最佳獎勵+自訂義獎勵的累積總合
        model_path = os.path.join(f"ckpt_test/{ROUND}",f"step_{timestep}_reward_{int(cumulative_custom_reward)}_distance_{int(distance)}.pth")
        torch.save(dqn.q_net.state_dict(), model_path)
        print(f"Model saved: {model_path}")

final_model_path = os.path.join(f"ckpt_test/{ROUND}", f"final_best_reward_{int(best_reward)}_best_distance_{int(best_distance)}.pth")
os.makedirs(f"ckpt_test/{ROUND}", exist_ok=True)
torch.save(dqn.q_net.state_dict(), final_model_path)
print(f"Final model saved: {final_model_path}")
env.close()

window_size = 10
averaged_rewards = [
    np.mean(reward_per_episode[i:i + window_size])
    for i in range(0, len(reward_per_episode), window_size)
]
averaged_distance = [
    np.mean(distance_per_timestep[i:i + window_size])
    for i in range(0, len(distance_per_timestep), window_size)
]

x_rewards = range(window_size, window_size * len(averaged_rewards) + 1, window_size)
plt.figure(figsize=(10, 6))
plt.plot(x_rewards, averaged_rewards, label='Averaged Rewards')
plt.title('Total Reward per Episode (Averaged)')
plt.xlabel('Timestep')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.savefig(f'ckpt_test/{ROUND}/total_reward_per_episode.png', dpi=300)

x_distance = range(window_size, window_size * len(averaged_distance) + 1, window_size)
plt.figure(figsize=(10, 6))
plt.plot(x_distance, averaged_distance, label='Averaged Distance')
plt.title('Distance per Episode (Averaged)')
plt.xlabel('Timestep')
plt.ylabel('Average Distance')
plt.legend()
plt.grid(True)
plt.savefig(f'ckpt_test/{ROUND}/Distance.png', dpi=300)