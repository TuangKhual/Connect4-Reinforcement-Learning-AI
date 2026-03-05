from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from connect4_env import connect4Env
import random

enemyPool = [] # gets a pool of the past 10 bots to train against
maxSize = 10

env = connect4Env()
check_env(env)

model = PPO("MlpPolicy", env, verbose = 1, learning_rate = 3e-4, n_steps=4096, batch_size=128, device="cpu")

updateInterval = 150_000 # every 150k steps a new generation is born
totalSteps = 45_000_000 # how many steps training you want to do

steps = 0
generation = 0

while steps < totalSteps:
    if enemyPool:
        chosenEnemy = PPO.load(random.choice(enemyPool))
        env.setEnemy(chosenEnemy)
    else:
        env.setEnemy(model) # first run it's training against random movies
    model.learn(total_timesteps=updateInterval, reset_num_timesteps=False)
    steps += updateInterval
    generation += 1
    model.save(f"connect4_gen{generation}") #Saves the generation 
    enemyPool.append(f"connect4_gen{generation}")
    print(f"Generation {generation} complete ({steps}/{totalSteps} steps)")
    if len(enemyPool) > maxSize: #removes when the pool is too big
        enemyPool.pop(0)

model.save("connect4_ai_largeV6")
print("Training complete!")