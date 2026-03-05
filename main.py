import pygame
from connect4 import connect4
from connect4 import connect4GUI
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
import numpy as np
import random

model = PPO.load("connect4_ai_largeV4") # loads whichever model you want

def aiMove(game, model, turn):
    obs = np.array(game.board, dtype = np.float32) * turn # if it's player 1 or 2 it changes the board accordingly
    action, _ = model.predict(obs, deterministic = True) # picks the best move
    if game.board[0][action] != 0: # if the column is full gets the first 
        col = []
        for i in range(7):
            if game.board[0][i] == 0:
                col.append(i)
        action = random.choice(col) # if the col is full just take a random col
    game.drop(int(action))


def main():
    pygame.init()
    game = connect4()
    gui = connect4GUI(game)

    running = True
    gameFinish = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False

            if event.type == pygame.MOUSEMOTION: # keep track of the position of the mouse when it moves
                gui.hoverMouse(event.pos)
            if event.type == pygame.MOUSEBUTTONDOWN and game.playerTurn == 1:    # if player 1 is the human and mouse is pressed
                if gui.button.collidepoint(event.pos): # If it hits the restart button
                    game.restart()
                    gui.winner = 0
                    gameFinish = False
                elif gameFinish == False: # If the game is not over yet drop it in the cik
                    col = event.pos[0] // gui.circleSize
                    game.drop(col)  
            if game.playerTurn == -1: # alternate turns
                aiMove(game, model, -1)

            winner = game.winCons() # checks fo winner after
            if winner != 0 and not gameFinish:
                gui.winner = winner
                gameFinish = True
                if winner == 1:
                    gui.p1Score += 1
                else:
                    gui.p2Score += 1
            
        gui.draw()    

    pygame.quit()

if __name__ == "__main__":
    main()


