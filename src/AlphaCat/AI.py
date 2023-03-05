import Game
import random

import numpy as np
import pandas as pd


class SmartAI:
    Q_table: pd.DataFrame = None  # All ai sharing a same Q-table

    def __init__(self, character: int, state=None, cache: str = None):
        self.game = state
        self.char = character
        self.eplis = 0.2
        self.l_rate = 0.1
        self.discount = 0.5

        if SmartAI.Q_table is None:
            if cache is None:
                SmartAI.Q_table = pd.DataFrame(np.zeros((3 ** (self.game.size ** 2), (self.game.size ** 2))))
                SmartAI.Q_table.index.name = "state"
            else:
                cache += f"_{self.game.size}_{self.game.max_len}.xlsx"
                print("load", cache)
                SmartAI.Q_table = pd.read_excel(cache).set_index("state")

    def get_action(self) -> tuple:
        action = None
        moves = self.game.get_avail_moves()

        if np.sum(self.game.board[moves]) != 0:
            _Q, action = max((self.Q_table[self.game.board][act], act) for act in moves)
        return action

    def move(self, train: bool = False):

        moves = self.game.get_avail_moves()
        if not moves:
            return

        avail = [self.game.get_1d_loc(m) for m in moves]
        state_o = self.game.get_state(self.char)
        max_v, best_move = max((SmartAI.Q_table.iloc[state_o][a], a) for a in avail)
        if train:
            if np.random.uniform() > self.eplis and max_v != 0:
                # exploitation

                for m in moves:
                    if self.game.check_win_fast(self.char, m):
                        self.game.move(self.char, m)
                        loc = self.game.get_1d_loc(m)
                        SmartAI.Q_table.iloc[state_o][loc] += self.l_rate * (5 - SmartAI.Q_table.iloc[state_o][loc])
                        return True

                # print(avail, move)
                reward, win = self.game.move(self.char, self.game.get_2d_loc(best_move))

                state_n = self.game.get_state(-self.char)

                SmartAI.Q_table.iloc[state_o][best_move] += self.l_rate * (reward
                                                                      - self.discount * max(SmartAI.Q_table.iloc[state_n])
                                                                      - SmartAI.Q_table.iloc[state_o][best_move])
                # print(state_o, move, reward)
                return win

            else:
                # exploration
                best_move = random.choice(moves)
                reward, win = self.game.move(self.char, best_move)
                state_n = self.game.get_state(self.char)

                SmartAI.Q_table.iloc[state_o][self.game.get_1d_loc(best_move)] += self.l_rate * (
                        reward
                        - self.discount * max(SmartAI.Q_table.iloc[state_n])
                        - SmartAI.Q_table.iloc[state_o][self.game.get_1d_loc(best_move)])

                return win

        else:

            max_v = np.max(SmartAI.Q_table.iloc[state_o][avail])
            best_move = np.random.choice([i for i in avail if SmartAI.Q_table.iloc[state_o][i] == max_v])
            reward, win = self.game.move(self.char, self.game.get_2d_loc(best_move))
            # self.game.display_grid()
            return win

    def rand_move(self):

        moves = self.game.get_avail_moves()

        if moves is not None:
            move = random.choice(moves)
            r, win = self.game.move(self.char, move)
            return win
        return False
