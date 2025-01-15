# Super Mario Bros RL Agent

## Repository Structure

- **DQN.py**: Contains the implementation of the Deep Q-Network (DQN) and Actor-Critic DQN (ACDQN) algorithms.
- **eval.py**: Used for evaluating the performance of the trained models.
- **find_best_model.py**: Script to find the best model by testing different skipframe values.
- **model.py**: Defines the neural network architecture for the DQN.
- **model_AC.py**: Defines the neural network architecture for the Actor-Critic model.
- **reward.py**: Contains various reward functions to incentivize the agent's behavior.
- **run.py**: Main training loop for the RL agent.
- **skipframe.py**: Wrapper to skip frames during training to speed up the process.
- **movement.py**: Defines the possible movements/actions for the agent.
- **utils.py**: Utility functions, such as preprocessing frames.

## Installation

### Clone the repository:

```bash
git clone https://github.com/livejiaquan/Super-mario-bros.git
cd Super-mario-bros
```

### Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the RL agent, run the `run.py` script. This script initializes the environment, sets up the DQN or ACDQN agent, and begins the training process.

```bash
python run.py
```

### Evaluation

To evaluate a trained model, use the `eval.py` script. Make sure to specify the path to the model weights in the script.

```bash
python eval.py
```

### Finding the Best Model

To find the best model by testing different skipframe values, run the `find_best_model.py` script.

```bash
python find_best_model.py
```

## Reward Functions

The `reward.py` file contains various functions to calculate rewards based on the agent's actions and the game's state. These functions include rewards for collecting coins, making vertical and horizontal movements, maintaining speed, and more.

## Custom Movements

The `movement.py` file defines different sets of movements/actions that the agent can perform. The `CUSTOM_MOVEMENT` set includes actions such as moving right, jumping, and running.

## Preprocessing

The `utils.py` file contains a function to preprocess game frames by converting them to grayscale and resizing them to 84x84 pixels.

## SkipFrame Wrapper

The `skipframe.py` file defines a wrapper to skip frames during training, which helps in speeding up the training process by reducing the number of frames the agent needs to process.

## Model Architectures

- **model.py**: Contains the architecture for the DQN model using convolutional and residual blocks.
- **model_AC.py**: Contains the architecture for the Actor-Critic model with shared convolutional layers and separate actor and critic heads.

## License

This project is licensed under the MIT License.

## Acknowledgments

This project uses the `gym_super_mario_bros` environment to simulate the Super Mario Bros game for training the RL agent. Special thanks to the authors of the libraries and tools used in this project.

For more information, please refer to the respective files in the repository.
