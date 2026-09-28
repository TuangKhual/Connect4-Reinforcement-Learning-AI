# Connect4-Reinforcement-Learning-AI
My first spin at using stable baselines3, gymnasium, and maskable PPO to train a connect4 agent through reinforcement learning and is trained against previous generations of itself.

How it works:
The connect 4 board is a 6x7 array, where 1 and -1 are representing the 2 players and 0 for an empty space. The agent receives the board as it's observation and using the masked PPO it chooses a non full column as its action.

Before the agent makes it's move it checks for any winning moves for itself and the opponent and blocks if needed. This gave it an extra edge in addition to the training.

I chose PPO as the reinforcement learning algorithm because it's what most people recommended for games like connect4. Basically what PPO does is plays the game, then checks the parameters you put and evaluates what rewards to give out. I chose this over minimax as I felt like I could learn more about machine learning and eventually learn pytorch from doing this project. 

The AI is trained in generations. After each set training interval, the generation is saved and is added to the enemy pool, so future generations can train off the previous. This ends up creating an agent that gets stronger generation after generation.

Requirements:
Gymnasium
Stable-Baselines3
SB3-Contrib / Maskable PPO
Numpy
Pygame

Install the dependencies and run with:
python main.py to play against the agent

or 

python train.py to train an agent for yourself



<img width="1224" height="1454" alt="image" src="https://github.com/user-attachments/assets/b5e6c4b8-143e-475c-94ed-2251c796e413" />


