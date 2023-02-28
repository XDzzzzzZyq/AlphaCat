import Game
import random


class SmartAI:
    game: Game.Game

    def __init__(self, character: int, state=None):
        self.game = state
        self.char = character

    def move(self):

        moves = self.game.get_avail_moves()
        if len(moves) > 0:
            self.game.move(self.char, random.sample(moves, 1)[0])