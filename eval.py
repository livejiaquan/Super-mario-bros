import numpy as np
import torch
import os
from tqdm import tqdm
import time

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from movement import CUSTOM_MOVEMENT
from utils import preprocess_frame
from model_AC import ActorCriticCNN 
# from model import CustomCNN
from DQN import ACDQN, DQN
from skipframe import SkipFrame

# ========== Config ===========
# 8
MODEL_PATH = '/Users/jiaquan/Desktop/JQ_NCKU/1131_Computer Vision In Deep Learning Implementation And Its Applications/Final_Project/Super-mario-bros/ckpt_test/final_best_reward_31266_best_distance_3161.pth'

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = SkipFrame(env, skip=8)

env = JoypadSpace(env, CUSTOM_MOVEMENT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBS_SHAPE = (1, 84, 84)
N_ACTIONS = len(CUSTOM_MOVEMENT)

VISUALIZE = True
TOTAL_EPISODES = 10

# ========== Initialize DQN =========== 
dqn = ACDQN(
    model=ActorCriticCNN, 
    state_dim=OBS_SHAPE,
    action_dim=N_ACTIONS,
    learning_rate=0.0001,  
    gamma=0.99,          
    epsilon=0.3,                   # 設為 0.0 表示完全利用當下的策略
    target_update=100,            # target [Q-net] 更新的頻率
    device=device
)

# ========== 載入模型權重 =========== 
if os.path.exists(MODEL_PATH):
    try:
        model_weights = torch.load(MODEL_PATH, map_location=device)
        dqn.q_net.load_state_dict(model_weights)
        dqn.q_net.eval()
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        raise
else:
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# ========== Evaluation Loop ===========
for episode in range(1, TOTAL_EPISODES + 1):
    state = env.reset()
    state = preprocess_frame(state)
    state = np.expand_dims(state, axis=0)
    state = np.expand_dims(state, axis=0)

    done = False
    total_reward = 0

    while not done:
        # Take action using the trained policy
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_logits, _ = dqn.q_net(state_tensor)
            action_probs = torch.softmax(action_logits, dim=1)
        
        # Epsilon-greedy action selection
        if np.random.rand() < dqn.epsilon:
            action = np.random.choice(N_ACTIONS)
        else:
            action = torch.argmax(action_probs, dim=1).item()

        next_state, reward, done, info = env.step(action)
        
        # Preprocess next state
        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)

        # Accumulate rewards
        total_reward += reward
        state = next_state

        if VISUALIZE:
            env.render()
            time.sleep(0.5)

    print(f"Episode {episode}/{TOTAL_EPISODES} - Total Reward: {total_reward}")

env.close()