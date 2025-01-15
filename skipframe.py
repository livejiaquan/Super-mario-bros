import gym

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        """跳幀 Wrapper，讓環境每隔 skip 幀進行一次動作選擇"""
        super(SkipFrame, self).__init__(env)
        self._skip = skip

    def step(self, action):
        """執行動作並跳過多幀"""
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)  # 執行動作
            total_reward += reward  # 累積獎勵
            if done:  # 如果遊戲結束，提前退出
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        """重置環境"""
        return self.env.reset(**kwargs)