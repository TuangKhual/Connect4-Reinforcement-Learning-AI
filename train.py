from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from connect4_env import connect4Env
from sb3_contrib import MaskablePPO
import random
import gymnasium as gym
import torch
import torch.nn as nn
import os


class Connect4CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.initial(x)
        x = x + self.res(x)
        return self.head(x)

env = connect4Env(poolSize = 10, minimaxDepths = [1], minimaxRatio = 0.1)
check_env(env)

policy_kwargs = dict(
    features_extractor_class=Connect4CNN,
    features_extractor_kwargs = dict(features_dim = 256),
)

model = MaskablePPO("CnnPolicy", env, verbose = 1, learning_rate = 1e-4, n_steps = 4096, batch_size = 512, ent_coef = 0.01, target_kl = 0.03, vf_coef = 0.5, gamma = 0.99, n_epochs = 10, device = "cpu", policy_kwargs = policy_kwargs)

updateInterval = 100_000
totalSteps = 10_000_000 

steps = 0
generation = 0
lastestPool = []
landmarkPool = []


LASTEST_SIZE = 10
LANDMARK_INTERVAL = 10
try:
    while steps < totalSteps:
        ratio = min(0.5, 0.1 + steps / 10_000_000) 
        if steps < 2_000_000:
            env.minimaxDepths = [1]
        elif steps < 4_000_000:
            env.minimaxDepths = [1, 2]
        elif steps < 6_000_000:
            env.minimaxDepths = [2, 3]
        elif steps < 8_000_000:
            env.minimaxDepths = [3, 4]
        else:
            env.minimaxDepths = [4, 5]
        
        model.learn(total_timesteps = updateInterval, reset_num_timesteps = False)
        steps += updateInterval
        generation += 1
        model.save(f"connect4_masked_gen_ONE_{generation}") 
        loadedModel = MaskablePPO.load(f"connect4_masked_gen_ONE_{generation}")

        lastestPool.append(loadedModel)
        if len(lastestPool) > LASTEST_SIZE:
            lastestPool.pop(0)

        if generation % LANDMARK_INTERVAL == 0:
            landmarkPool.append(loadedModel)

        
        env.enemyPool = lastestPool + landmarkPool
        print(f"Generation {generation} complete ({steps}/{totalSteps} steps)")

    model.save("connect4_masked_ai_large_Compare")
    print("Training complete!")

finally:
    os.system("shutdown -h +1")   # needs root; cancel with: shutdown -c
