import Game
import random

import numpy as np
import pandas as pd


class SmartAI:
    Q_table: pd.DataFrame = None  # All ai sharing a same Q-table

    def __init__(self, character: int, state=None, cache: str = None):
        self.game = state
        self.char = character
        self.eplis = 0.3
        self.l_rate = 0.1
        self.discount = 0.5

        if SmartAI.Q_table is None:
            if cache is None:
                SmartAI.Q_table = pd.DataFrame(np.zeros((3 ** (self.game.size ** 2), (self.game.size ** 2))))
            else:
                print("load", cache)
                SmartAI.Q_table = pd.read_excel(cache)

    def move(self, train: bool = False):

        moves = self.game.get_avail_moves()
        if not moves:
            return

        avail = [self.game.get_1d_loc(m) for m in moves]
        state_o = self.game.get_state(self.char)
        if train:
            if np.random.uniform() > self.eplis:
                # exploitation

                max_v = np.max(SmartAI.Q_table.iloc[state_o][avail])
                move = np.random.choice([i for i in avail if SmartAI.Q_table.iloc[state_o][i] == max_v])
                # print(avail, move)
                reward, win = self.game.move(self.char, self.game.get_2d_loc(move))

                state_n = self.game.get_state(-self.char)

                SmartAI.Q_table.iloc[state_o][move] += self.l_rate * (reward
                                                                      - self.discount * max(SmartAI.Q_table.iloc[state_n])
                                                                      - SmartAI.Q_table.iloc[state_o][move])
                # print(state_o, move, reward)
                return win

            else:
                # exploration
                move = random.choice(moves)
                reward, win = self.game.move(self.char, move)

                SmartAI.Q_table.iloc[state_o][self.game.get_1d_loc(move)] += self.l_rate * (
                        reward - SmartAI.Q_table.iloc[state_o][self.game.get_1d_loc(move)])

                return win

        else:

            max_v = np.max(SmartAI.Q_table.iloc[state_o][avail])
            move = np.random.choice([i for i in avail if SmartAI.Q_table.iloc[state_o][i] == max_v])
            reward, win = self.game.move(self.char, self.game.get_2d_loc(move))
            #self.game.display_grid()
            return win

    def rand_move(self):

        moves = self.game.get_avail_moves()

        if moves is not None:
            move = random.choice(moves)
            r, win = self.game.move(self.char, move)
            return win
        return False
