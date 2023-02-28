import Game
import random

import numpy as np
import pandas as pd


class SmartAI:

    Q_table = pd.DataFrame(np.zeros((3**9, 9)))

    def __init__(self, character: int, state=None):
        self.game = state
        self.char = character
        self.eplis = 0.2
        self.l_rate = 0.1
        self.discount = 0.9

    def move(self):

        moves = self.game.get_avail_moves()
        if not moves:
            return

        state_o = self.game.get_state(self.char)
        if np.random.uniform() > self.eplis:
            # exploitation
            avail = [self.game.get_1d_loc(m) for m in moves]
            max_v = np.max(SmartAI.Q_table.iloc[state_o][avail])
            move = np.random.choice(np.arange(self.game.size**2)[SmartAI.Q_table.iloc[state_o] == max_v])
            self.game.move(self.char, self.game.get_2d_loc(move))

        else:
            # exploration
            self.game.move(self.char, random.choice(moves))
        state_n = self.game.get_state(self.char)