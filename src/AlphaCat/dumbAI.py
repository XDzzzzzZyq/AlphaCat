import Game
import random


class DumbAI:

    def __init__(self, character: int, state=None):
        self.game = state
        self.char = character

    def move(self) -> bool:
        moves = self.game.get_avail_moves()

        # if itself could win
        for m in moves:
            if self.game.check_win_fast(self.char, m):
                self.game.move(self.char, m)
                return True
        # if enemy could win
        for m in moves:
            if self.game.check_win_fast(-self.char, m):
                self.game.move(self.char, m)
                return False

        # check corner
        corner = [(0, 0), (2, 0), (0, 2), (2, 2)]
        corner = [m for m in moves if m in corner]
        if len(corner) > 0:
            self.game.move(self.char, random.choice(corner))
            return False

        # check center
        if self.game.board[1, 1] == 0:
            self.game.move(self.char, (1, 1))
            return False

        if len(moves) > 0:
            self.game.move(self.char, random.choice(moves))
            return False

    def rand_move(self):

        moves = self.game.get_avail_moves()

        if moves is not None:
            move = random.choice(moves)
            r, win = self.game.move(self.char, move)
            return win
        return False
