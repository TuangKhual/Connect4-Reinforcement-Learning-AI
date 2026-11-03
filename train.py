from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from connect4_env import connect4Env
from sb3_contrib import MaskablePPO
import random
import gymnasium as gym
import torch
import torch.nn as nn

class Connect4CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)

env = connect4Env(poolSize = 10, minimaxDepth = 5, minimaxRatio = 0.2)
check_env(env)

policy_kwargs = dict(
    features_extractor_class=Connect4CNN,
    features_extractor_kwargs = dict(features_dim = 128),
)

model = MaskablePPO("CnnPolicy", env, verbose = 1, learning_rate = 1e-4, n_steps = 4096, batch_size = 512, ent_coef = 0.05, device = "cpu", policy_kwargs = policy_kwargs)

updateInterval = 150_000
totalSteps = 7_500_000 

steps = 0
generation = 0

while steps < totalSteps:
    if steps >= 2_000_000:
        env.minimaxRatio = 0.4
    model.learn(total_timesteps = updateInterval, reset_num_timesteps = False)
    steps += updateInterval
    generation += 1
    model.save(f"connect4_masked_gen_ONE_{generation}") #Saves the generation
    env.enemyPool.append(MaskablePPO.load(f"connect4_masked_gen_ONE_{generation}"))
    print(f"Generation {generation} complete ({steps}/{totalSteps} steps)")
    if len(env.enemyPool) > env.poolSize: #removes when the pool is too big
        env.enemyPool.pop(0)


model.save("connect4_masked_ai_largeV10")
print("Training complete!")

## import os
## os.system("shutdown -h +1")   # needs root; cancel with: shutdown -c
