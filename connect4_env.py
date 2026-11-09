import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from gymnasium import spaces
from connect4 import connect4
from sb3_contrib import MaskablePPO
from minimaxPy import minimaxAi
import random

class connect4Env(gym.Env):
    def __init__(self, poolSize = 10, minimaxDepths = None, minimaxRatio = 0.2):
        self.enemyPool = []
        self.game = connect4()
        self.enemy = None
        self.observation_space = spaces.Box(low = -1, high = 1, shape=(1,6,7), dtype = np.float32) # the connect4 square (channels, height, width)
        self.action_space = spaces.Discrete(7) 
        self._minimaxAis = {}
        self.minimaxDepths = minimaxDepths or [1]
        self.minimaxRatio = minimaxRatio
        self.poolSize = poolSize

    @property
    def minimaxDepths(self):
        return self._minimaxDepths

    @minimaxDepths.setter
    def minimaxDepths(self, depths):
        self._minimaxDepths = depths
        for dep in depths:
            if dep not in self._minimaxAis:
                self._minimaxAis[dep] = minimaxAi(dep)

    def action_masks(self):
        return np.array([self.game.board[0][c] == 0 for c in range(7)])

    def findWinningMove(self, game, player):
        won = False
        for col in range(7):
            if game.board[0][col] != 0:  # move if full
                continue            
            
            currentTurn = game.playerTurn  # saves the actual turn so you can check if the opponent has a winning move too
            game.playerTurn = player
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

    def chooseEnemy(self):
        if not self.enemyPool:
            self.enemy = None
        elif random.random() < self.minimaxRatio:
            self.enemy = "minimax"
        else:
            self.enemy = random.choice(self.enemyPool)

    def reset(self, seed=None, options=None):
        super().reset(seed = seed)
        self.game.restart()
        self.chooseEnemy()

        self.agentTurn = random.choice([-1, 1]) # I want the ai to train as both player 1 and player 2

        if self.agentTurn == -1: # if the agent gets player 2 has the enemy move first
            self.enemyMove() 

        return self.getObsSpace(), {}
    
    def step(self, action):
        reward = 0.0
        self.game.drop(action)

        winner = self.game.winCons() # checks for win

        if winner != 0:
            if winner == self.agentTurn: #checks to see it wins
                reward += 1.0
            return self.getObsSpace(), reward, True, False, {}

        if all(self.game.board[0][i] != 0 for i in range(7)):
            return self.getObsSpace(), reward, True, False, {}

        self.enemyMove()
        winner = self.game.winCons()

        if winner != 0:
            if winner != self.agentTurn: #checks to see it wins
                reward -= 1.0
            return self.getObsSpace(), reward, True, False, {}
        
        if all(self.game.board[0][i] != 0 for i in range(7)): #checks to see if the board is full then its a tie
            return self.getObsSpace(), reward, True, False, {}
            
        return self.getObsSpace(), reward, False, False, {}

    def enemyMove(self):
        col = []
        for i in range(7):
            if self.game.board[0][i] == 0: #checks if the top is full 
                col.append(i)
        if not col:
            return
        if self.enemy == "minimax":
            depth = random.choice(self.minimaxDepths)
            action = self._minimaxAis[depth].pyAction(self.game.board, -self.agentTurn)
            self.game.drop(action)
        elif self.enemy is None: # the first time training
            self.game.drop(random.choice(col)) #drops in that random col
        else:
            aiPiece = sum(row.count(self.agentTurn) for row in self.game.board) # added the same for the enemy ai
            if aiPiece == 0 and self.game.board[0][3] == 0:
                self.game.drop(3)
                return

            winningMove = self.findWinningMove(self.game, -self.agentTurn)
            if winningMove != None:
                self.game.drop(winningMove)
                return

            blockWinningMove = self.findWinningMove(self.game, self.agentTurn)
            if blockWinningMove != None:
                self.game.drop(blockWinningMove)
                return
        
            obs = (np.array(self.game.board, dtype = np.float32) * -self.agentTurn).reshape(1, 6, 7) #flips board view in case the ai is player as player 2
            action, _ = self.enemy.predict(obs, action_masks = self.action_masks(), deterministic = False)
            if self.game.board[0][int(action)] != 0:
                col = [i for i in range(7) if self.game.board[0][i] == 0]
                if not col:
                    return
                action = random.choice(col)
            self.game.drop(int(action))

    def getObsSpace(self):
        board = np.array(self.game.board, dtype=np.float32) * self.agentTurn
        return board.reshape(1, 6, 7)

