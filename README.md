# Connect4-Reinforcement-Learning-AI
My first spin at using stable baselines3 and gymnasium to train a connect4 agent that can hopefully beat me.

Make sure to have numpy, pygame, stable baselines3, and gymnasium

Make sure to download all the files (You don't need the largeV4.zip) and to train just put the total steps and increments you want and run the file. Make sure to include the name of your ai in the main model and run main to play.

I chose PPO as the reinforcement learning algorithm because it's what most people recommended for games like connect4. Basically what PPO does is plays the game, then checks the parameters you put and evaluates what rewards to give out. I chose this over minimax as I felt like I could learn more about machine learning and eventually learn pytorch from doing this project. 

The observation space it chooses its action from is the same as a connect 4 board. At first I made the rewards for losing -1.5, winning +1.0, and drawing +0.2. I was looking at the training data and it seemed like games were ending quickly, so I tried to reward the AI for making more moves and punished it more for picking invalid moves. The setup I ended up with is this: Invalid move: -1.0 Losing: -1.0 Winning: +1.0 and Draw: 0.0. 

Things to work on: I’ll need to train a bigger agent and I’m thinking about adding someone called maskablePPO. That should fix the invalid move selection I’m having and should help the training.
<img width="1224" height="1454" alt="image" src="https://github.com/user-attachments/assets/b5e6c4b8-143e-475c-94ed-2251c796e413" />


