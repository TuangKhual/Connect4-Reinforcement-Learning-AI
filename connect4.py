import pygame

class connect4:
    rows = 6
    cols = 7

    def __init__(self):
        self.board = []
        for row in range(self.rows): # builds 2d array
            rows = []
            for col in range(self.cols):
                rows.append(0)
            self.board.append(rows)

        self.playerTurn = 1

    def drop(self, col):
        for row in range(self.rows -1, -1, -1):   # drops to the lowest row
            if self.board[row][col] == 0:
                self.board[row][col] = self.playerTurn
                self.playerTurn = self.playerTurn * -1 # changes player turn
                return True
        return False
    
    def restart(self):
        for row in range(self.rows):
            for col in range(self.cols):
                self.board[row][col] = 0 # sets everything to 0

        self.playerTurn = 1
    
    def winCons(self):
        for row in range(self.rows): #checking horizontal
            for col in range(self.cols - 3):
                piece = self.board[row][col]
                if piece == 0:
                    continue
                if self.board[row][col+1] == piece:
                    if self.board[row][col+2] == piece:
                        if self.board[row][col+3] == piece:
                            return piece
        
        for col in range(self.cols): #checking vertical
            for row in range(self.rows - 3):
                piece = self.board[row][col]
                if piece == 0:
                    continue
                if self.board[row+1][col] == piece:
                    if self.board[row+2][col] == piece:
                        if self.board[row+3][col] == piece:
                            return piece
        
        for row in range(self.rows - 3): # checking diagonal down right
            for col in range(self.cols - 3):
                piece = self.board[row][col]
                if piece == 0:
                    continue
                if self.board[row+1][col+1] == piece:
                    if self.board[row+2][col+2] == piece:
                        if self.board[row+3][col+3] == piece:
                            return piece
        
        for row in range(self.rows - 3): # checking down left
            for col in range(3, self.cols):
                piece = self.board[row][col]
                if piece == 0:
                    continue
                if self.board[row+1][col-1] == piece:
                    if self.board[row+2][col-2] == piece:
                        if self.board[row+3][col-3] == piece:
                            return piece
        return 0

    
class connect4GUI:
    circleSize = 100   
    screenWidth = circleSize * 7
    screenHeight = circleSize * 6 + 200

    colors = {
        0 : (255, 255, 255), # color for empty circle
        1 : (255, 0, 0), #player 1 color
        -1 : (0, 0, 255), # player 2 color
        2 : (255, 246, 154) # highlight mouse
    }

    

    def __init__(self, connect4Game):
        self.game = connect4Game
        self.screen = pygame.display.set_mode((self.screenWidth, self.screenHeight))
        self.hoverCol = None
        self.hoverRow = None
        self.button = pygame.Rect(self.screenWidth // 2 - 50, self.screenHeight - 80, 100, 60)
        self.winner = 0
        self.p1Score = 0
        self.p2Score = 0

    def draw(self):
        self.screen.fill((135, 135, 135))


        font = pygame.font.Font(None, 40)
        score_text = f"P1: {self.p1Score}        P2: {self.p2Score}" # writes scores to screen
        text = font.render(score_text, True, (0, 200, 0))
        
        pygame.draw.rect(self.screen, (200, 0, 0), self.button)
        buttonText = font.render("Restart", True, (255, 255, 255)) # writes the restart button
        textOnButton = buttonText.get_rect(center=self.button.center)

        self.screen.blit(buttonText, textOnButton)
        self.screen.blit(text, (100, self.game.rows * self.circleSize + 100))

        if self.winner == 1:
            winText = font.render("Player 1 wins!", True, (200, 0, 0))
            self.screen.blit(winText, (self.screenWidth // 2 + 100, self.screenHeight - 100)) # winner text

        elif self.winner == -1:
            winText = font.render("Player 2 wins!", True, (200, 0, 0))
            self.screen.blit(winText, (self.screenWidth // 2 + 100, self.screenHeight - 100))

        if self.hoverCol is not None and 0 <= self.hoverCol < self.game.cols:
            for i in range(self.game.rows - 1, -1, -1):
                if self.game.board[i][self.hoverCol] == 0: # Highlights the lowest rom in the column 
                    self.hoverRow = i
                    break

        for row in range(self.game.rows):
            for col in range(self.game.cols):
                piece = self.game.board[row][col] # gets that spots number could be a -1 0 or 1
                if self.hoverRow is not None and self.hoverCol is not None and row == self.hoverRow and col == self.hoverCol and self.game.board[row][col] == 0:
                    color = self.colors[2] # makes color yellow if it's the hovered row and col
                else:
                    color = self.colors[piece] # else it uses the number value inside the board
                pygame.draw.circle(self.screen, color, (col * self.circleSize + 50, row * self.circleSize + 50), 40)
        pygame.display.flip()

    def hoverMouse(self, pos):
        self.hoverCol = pos[0] // self.circleSize # Gets the position and divides it to see which col section it's in

