# Env state
# info = {
#     "x_pos",  # (int) The player's horizontal position in the level.
#     "y_pos",  # (int) The player's vertical position in the level.
#     "score",  # (int) The current score accumulated by the player.
#     "coins",  # (int) The number of coins the player has collected.
#     "time",   # (int) The remaining time for the level.
#     "flag_get",  # (bool) True if the player has reached the end flag (level completion).
#     "life"   # (int) The number of lives the player has left.
# }


#===============to do===============================請自定義獎勵函數 至少7個(包含提供的)

def calculate_coin_reward(info, reward, prev_info):
    """
    Calculate additional rewards for collecting coins during the game.
    
    Args:
        info (dict): Current game information.
        reward (float): Current reward from the environment.
        prev_info (dict): Previous game information.

    Returns:
        float: Updated total reward.
    """
    total_reward = reward
    total_reward += (info['coins'] - prev_info['coins']) * 50  # Grant 50 points for each coin collected.
    return total_reward

def calculate_vertical_movement_reward(info, reward, prev_info):
    """
    Encourage the player to make vertical movements, such as jumping or climbing.

    Args:
        info (dict): Current game information.
        reward (float): Current reward from the environment.
        prev_info (dict): Previous game information.

    Returns:
        float: Updated total reward.
    """
    total_reward = reward
    y_offset_change = info['y_pos'] - prev_info['y_pos']
    if y_offset_change > 0:
        total_reward += 3  # Bonus for upward movement.
    elif y_offset_change < 0:
        total_reward += 1  # Smaller bonus for downward movement.
    return total_reward

def calculate_horizontal_movement_reward(info, reward, prev_info):
    """
    Reward the player for forward progress and penalize for stalling or moving backward.

    Args:
        info (dict): Current game information.
        reward (float): Current reward from the environment.
        prev_info (dict): Previous game information.

    Returns:
        float: Updated total reward.
    """
    total_reward = reward
    x_offset_change = info['x_pos'] - prev_info['x_pos']
    if x_offset_change > 2:
        total_reward += 3  # Bonus for significant forward movement.
    elif x_offset_change < -2:
        total_reward += 1  # Smaller bonus for backward movement.
    else:
        total_reward -= 5  # Penalty for minimal or no movement.
    return total_reward

def calculate_speed_based_reward(info, reward, distance):
    """
    Reward the player for maintaining a high speed by measuring progress over time.

    Args:
        info (dict): Current game information.
        reward (float): Current reward from the environment.
        distance (int): Previous x-axis position.

    Returns:
        tuple: Updated total reward and updated distance.
    """
    total_reward = reward
    if info['x_pos'] > distance:
        time_factor = 1 + info['time'] / 100  # Scale reward based on remaining time.
        total_reward += 10 * (info['x_pos'] - distance) / time_factor
        distance = info['x_pos']
    return total_reward, distance

def calculate_goal_completion_reward(info, reward):
    """
    Provide a reward multiplier if the player reaches the goal flag.

    Args:
        info (dict): Current game information.
        reward (float): Current reward from the environment.

    Returns:
        float: Updated total reward.
    """
    total_reward = reward
    if info['flag_get']:
        total_reward *= 1.2  # Increase total reward by 20% if the flag is reached.
    return total_reward

def calculate_score_reward(info, reward, prev_info):
    """
    Reward the player for any increase in the game score.

    Args:
        info (dict): Current game information.
        reward (float): Current reward from the environment.
        prev_info (dict): Previous game information.

    Returns:
        float: Updated total reward.
    """
    total_reward = reward
    score_increase = info['score'] - prev_info['score']
    if score_increase > 0:
        total_reward += score_increase
    return total_reward

def apply_death_penalty(info, reward, prev_info):
    """
    Penalize the player heavily for losing a life.

    Args:
        info (dict): Current game information.
        reward (float): Current reward from the environment.
        prev_info (dict): Previous game information.

    Returns:
        float: Updated total reward.
    """
    total_reward = reward
    if prev_info['life'] > info['life']:
        total_reward -= 2000  # Heavy penalty for losing a life.
    return total_reward

def calculate_altitude_bonus(info, reward, max_y=10):
    """
    Provide rewards for achieving higher vertical positions.

    Args:
        info (dict): Current game information.
        reward (float): Current reward from the environment.
        max_y (int): Maximum altitude value to cap the reward.

    Returns:
        float: Updated total reward.
    """
    total_reward = reward
    if info['y_pos'] > 2:
        total_reward += min(info['y_pos'], max_y) * 2  # Reward proportional to altitude, capped at max_y.
    return total_reward

def calculate_survival_time_reward(info, reward, prev_info):
    """
    Reward the player for staying alive longer by adding a bonus based on elapsed time.

    Args:
        info (dict): Current game information.
        reward (float): Current reward from the environment.
        prev_info (dict): Previous game information.

    Returns:
        float: Updated total reward.
    """
    total_reward = reward
    time_passed = prev_info['time'] - info['time']
    if time_passed > 0:
        total_reward += time_passed * 3  # Reward for time spent alive.
    return total_reward

def apply_stagnation_penalty(reward):
    """
    Apply a penalty if the player exhibits prolonged stagnation (e.g., minimal movement).

    Args:
        reward (float): Current reward from the environment.

    Returns:
        float: Updated total reward with stagnation penalty applied.
    """
    total_reward = reward
    return total_reward - 1500  # Heavy penalty for stagnation.
