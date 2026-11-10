from connect4 import connect4
from sb3_contrib import MaskablePPO
from minimaxPy import minimaxAi
import numpy as np
import random

def load_model(path):
    try:
        return MaskablePPO.load(path, device="cpu"), "maskable"
    except Exception:
        raise ValueError(f"Could not load {path}")

def get_obs_space(game, player):
    board = np.array(game.board, dtype=np.float32) * player
    return board.reshape(1, 6, 7)

def action_masks(game):
    return np.array([game.board[0][c] == 0 for c in range(7)])

def check_draw(game):
    if all(game.board[0][i] != 0 for i in range(7)):
        return -1
    return 1

def move(game, model, player):
    if check_draw(game) == -1:
        return
    obs = get_obs_space(game, player)
    action, _ = model.predict(obs, action_masks = action_masks(game), deterministic = False)
    action = int(action)
    game.drop(action)

def minimax_move(game, ai, player):
    if check_draw(game) == -1:
        return
    action = ai.pyAction(game.board, player)
    game.drop(action)

def game_loop(p1, p2) :
    game = connect4()
    players = {1: p1, -1: p2}

    while True:
        curr_turn = game.playerTurn
        curr_player = players[curr_turn]

        if curr_player[1] == "minimax":
            minimax_move(game, curr_player[0], curr_turn)
        else:
            model = curr_player
            move(game, model[0], curr_turn)

        winner = game.winCons()
        if winner != 0:
            return winner
        if check_draw(game) == -1:
            return 0

def sim_match(p1, p2, num_games = 100):
    results = {1: 0, -1: 0, 0: 0}

    for i in range(num_games):
        if i % 2 == 0:
            winner = game_loop(p1, p2)
        else:
            winner = game_loop(p2, p1)
            if winner != 0:
                winner *= -1

        results[winner] += 1
    return results

def model(choice, minimax_depth = None):
    if choice.lower().startswith("minimax"):
        split_choice = choice.split()
        if len(split_choice) > 1:
            depth = int(split_choice[1])
        ai = minimaxAi(depth)
        return (ai, "minimax")
    else:
        model, model_type = load_model(choice)
        return (model, model_type)

if __name__ == "__main__":
    print("Enter model name or minimax + depth (ex: minimax 3)")

    p1_path = input("Player 1: ").strip()
    p2_path = input("Player 2: ").strip()

    num_games = int(input("Number of games to simulate (Default is 100): ").strip() or "100")

    p1 = model(p1_path)
    p2 = model(p2_path)

    print("Running simulation")
    results = sim_match(p1, p2, num_games)

    print(f"Player 1 wins: {results[1]}")
    print(f"Player 2 wins: {results[-1]}")
    print(f"Draws: {results[0]}")