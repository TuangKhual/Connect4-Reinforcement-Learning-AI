import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from gymnasium import spaces
from connect4 import connect4
from sb3_contrib import MaskablePPO
import random

class connect4Env(gym.Env):
    def __init__(self):
        self.game = connect4()
        self.enemy = None
        self.observation_space = spaces.Box(low = -1, high = 1, shape=(6,7), dtype = np.float32) # the connect4 square
        self.action_space = spaces.Discrete(7) # can only do 0 to 6 aka 7 columns
        self.first_move = True

    def action_masks(self):
        return np.array([self.game.board[0][c] == 0 for c in range(7)])

    def findWinningMove(self, game, player):
        won = False
        currentTurn = game.playerTurn  # saves the actual turn so you can check if the opponent has a winning move too
        game.playerTurn = player
        for col in range(7):
            if game.board[0][col] != 0:  # move if full
                continue
            game.drop(col)
            if (game.winCons() == player): 
                won = True

            for row in range(6):   # delete the move that was just made
                if game.board[row][col] != 0:
                    game.board[row][col] = 0
                    break
            game.playerTurn = currentTurn
            if won:
                return col
        return None

    def setEnemy(self, model):
        model.save("tempEnemy")    # saves enemy so it can be loaded
        self.enemy = MaskablePPO.load("tempEnemy", device="cpu") 

    def reset(self, seed=None, options=None):
        super().reset(seed = seed)
        self.game.restart()

        self.agentTurn = random.choice([-1, 1]) # I want the ai to train as both player 1 and player 2

        if self.agentTurn == -1: # if the agent gets player 2 has the enemy move first
            self.enemyMove() 
            self.first_move = False
        else:
            self.first_move = True
        return self.getObsSpace(), {}
    
    def step(self, action):
        reward = 0.0
        winningMove = self.findWinningMove(self.game, self.agentTurn) # if finds the winning move
        if winningMove != None:
            self.game.drop(winningMove)
            return self.getObsSpace(), 1.0, True, False, {} # ends the game

        blockWinningMove = self.findWinningMove(self.game, -self.agentTurn) #blocks a winning move
        if blockWinningMove != None:
            self.game.drop(blockWinningMove)
            reward += 0.10 # gets alittle reward
        else:
            self.game.drop(action)
            if (self.first_move and action == 3):
                reward += 0.05

        winner = self.game.winCons() # checks for win

        if winner != 0:
            if winner == self.agentTurn: #checks to see it wins
                reward = 1.0
            else:
                reward = -1.0
            return self.getObsSpace(), reward, True, False, {}

        if all(self.game.board[0][i] != 0 for i in range(7)):
            return self.getObsSpace(), 0.0, True, False, {}

        self.enemyMove()
        winner = self.game.winCons()

        if winner != 0:
            if winner == self.agentTurn: #checks to see it wins
                reward = 1.0
            else:
                reward = -1.0
            return self.getObsSpace(), reward, True, False, {}
        
        if all(self.game.board[0][i] != 0 for i in range(7)): #checks to see if the board is full then its a tie
            return self.getObsSpace(), 0.0, True, False, {}
            
        return self.getObsSpace(), 0.0, False, False, {}

    def enemyMove(self):
        col = []
        for i in range(7):
            if self.game.board[0][i] == 0: #checks if the top is full 
                col.append(i)
        if not col:
            return
        if self.enemy is None: # the first time training
            if (self.first_move):
                self.game.drop(3) #drops in middle because that's the best first move
                self.first_move = False
            else:
                self.game.drop(random.choice(col)) #drops in that random col
        else:
            obs = np.array(self.game.board, dtype = np.float32) * -self.agentTurn #flips board view in case the ai is player as player 2
            action, _ = self.enemy.predict(obs, action_masks = self.action_masks(), deterministic = False)
            if self.game.board[0][int(action)] != 0:
                col = [i for i in range(7) if self.game.board[0][i] == 0]
                if not col:
                    return
                action = random.choice(col)
            self.game.drop(int(action))

    def getObsSpace(self):
        board = np.array(self.game.board, dtype=np.float32) * self.agentTurn
        return board

