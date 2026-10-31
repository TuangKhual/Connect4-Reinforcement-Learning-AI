#include <limits>
using namespace std;

const int ROW_COUNT = 6;
const int COL_COUNT = 7;
const int DRAW = 50;
int board[ROW_COUNT][COL_COUNT];

int dropPiece(int col, int piece){
    for (int i = 5; i >= 0; i--){
        if (board[i][col] == 0){
            board[i][col] = piece;
            return i;
        }
    }
    return -1;
}

int window(int p1, int p2, int p3, int p4, int piece){
    int enemy = -piece;
    int pieceCount = 0;
    int emptyCount = 0;
    int enemyCount = 0;
    int score = 0;
    int pieceArray[4] = {p1, p2, p3, p4};
    for (int i = 0; i < 4; ++i){
        if (pieceArray[i] == piece){
            pieceCount++;
        }
        else if (pieceArray[i] == enemy){
            enemyCount++;
        }
        else{
            emptyCount++;
        }
    }
    if (pieceCount == 4){
        score += 100;
    }
    else if (pieceCount == 3 && emptyCount == 1){
        score += 5;
    }
    else if (pieceCount == 2 && emptyCount == 2){
        score += 2;
    }
    if (enemyCount == 3 && emptyCount == 1){
        score -= 5;
    }
    else if (enemyCount == 2 && emptyCount == 2){
        score -= 2;
    }
    return score;
}

int scoreCount(int piece){
    int score = 0;
    // horizontal check
    for (int row = 0; row < ROW_COUNT; ++row){
        for (int col = 0; col < 4; ++col){
            score += window(board[row][col], board[row][col + 1], board[row][col + 2], board[row][col + 3], piece);
        }
    }

    for (int col = 0; col < COL_COUNT; ++col){
        for (int row = 0; row < 3; ++row){
            score += window(board[row][col], board[row + 1][col], board[row + 2][col], board[row + 3][col], piece);
        }
    }
    // diagonal check
    for (int row = 0; row < 3; ++row){
        for (int col = 0; col < 4; ++col){
            score += window(board[row][col], board[row + 1][col + 1], board[row + 2][col + 2], board[row + 3][col + 3], piece);
        }
    }

    for (int row = 0; row < 3; ++row){
        for (int col = 3; col < COL_COUNT; ++col){
            score += window(board[row][col], board[row + 1][col - 1], board[row + 2][col - 2], board[row + 3][col - 3], piece);
        }
    }
    return score;
}

int winCon(){
    // horizontal check
    for (int row = 0; row < ROW_COUNT; ++row){
        for (int col = 0; col < 4; ++col){
            int p = board[row][col];
            if (p != 0){
                if (board[row][col] == p && board[row][col + 1] == p && board[row][col + 2] == p && board[row][col + 3] == p){
                    return p;
                }
            }
        }
    }

    for (int col = 0; col < COL_COUNT; ++col){
        for (int row = 0; row < 3; ++row){
            int p = board[row][col];
            if (p != 0){
                if (board[row][col] == p && board[row + 1][col] == p && board[row + 2][col] == p && board[row + 3][col] == p){
                    return p;
                }
            }
        }
    }
    // diagonal check
    for (int row = 0; row < 3; ++row){
        for (int col = 0; col < 4; ++col){
            int p = board[row][col];
            if (p != 0){
                if (board[row][col] == p && board[row + 1][col + 1] == p && board[row + 2][col + 2] == p && board[row + 3][col + 3] == p){
                    return p;
                }
            }
        }
    }

    for (int row = 0; row < 3; ++row){
        for (int col = 3; col < COL_COUNT; ++col){
            int p = board[row][col];
            if (p != 0){
                if (board[row][col] == p && board[row + 1][col - 1] == p && board[row + 2][col - 2] == p && board[row + 3][col - 3] == p){
                    return p;
                }
            }
        }
    }
    bool full = true;
    for (int col = 0; col < COL_COUNT; ++col){
        if (board[0][col] == 0){
            full = false;
            break;
        }
    }
    if (full){
        return DRAW;
    }
    return 0;
}

int minimax(int depth, int alpha, int beta, int maximizingPlayer, int piece, int *bestCol){
    int winner = winCon();
    if (winner == piece){
        return 100000;
    }
    if (winner == -piece){
        return -100000;
    }
    if (winner == DRAW){
        return 0;
    }
    if (depth == 0){
        return scoreCount(piece);
    }

    if (maximizingPlayer){
        int value = -200000;
        for (int col = 0; col < COL_COUNT; ++col){
            if (board[0][col] != 0){
                continue;
            }
            int row = dropPiece(col, piece);
            int tempCol;
            int score = (minimax(depth - 1, alpha, beta, 0, piece, &tempCol));
            board[row][col] = 0;
            if (score > value){
                value = score;
                *bestCol = col;
            }
            if (value > alpha) {
                alpha = value;
            }
            if (alpha >= beta){
                break;
            }
        }
        return value;
    }
    else {
        int value = 200000;
        for (int col = 0; col < COL_COUNT; ++col){
            if (board[0][col] != 0){
                continue;
            }
            int row = dropPiece(col, -piece);
            int tempCol;
            int score = (minimax(depth - 1, alpha, beta, 1, piece, &tempCol));
            board[row][col] = 0;
            if (score < value){
                value = score;
                *bestCol = col;
            }
            if (value < beta) {
                beta = value;
            }
            if (alpha >= beta){
                break;
            }
        }
        return value;
    }
}

extern "C" int pyAction(int flattenBoard[42], int depth, int piece){
    for (int row = 0; row < ROW_COUNT; ++row){
        for (int col = 0; col < COL_COUNT; ++col){
            board[row][col] = flattenBoard[row * 7 + col];
        }
    }
    int col = 3;
    minimax(depth, -200000, 200000, 1, piece, &col);
    return col;
}