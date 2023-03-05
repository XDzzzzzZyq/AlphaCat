import Game
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

from MLP import DeepQMLP


class AI:

    def __init__(self, character: int, state=None):
        self.game = state
        self.char = character
        self.eplis = 0.2
        self.l_rate = 0.1
        self.discount = 0.5

    def move(self, train: bool = False) -> bool:
        pass

    def get_action(self) -> tuple:
        pass

    def rand_move(self):
        moves = self.game.get_avail_moves()

        if moves is not None:
            move = random.choice(moves)
            r, win = self.game.move(self.char, move)
            return win
        return False


class SmartAI(AI):
    Q_table: pd.DataFrame = None  # All ai sharing a same Q-table

    def __init__(self, character: int, state=None, cache: str = None):
        super().__init__(character, state)
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

    def move(self, train: bool = False) -> bool:

        moves = self.game.get_avail_moves()
        if not moves:
            return False

        avail = [self.game.get_1d_loc(m) for m in moves]
        state_o = self.game.get_int_state(self.char)
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

                state_n = self.game.get_int_state(-self.char)

                SmartAI.Q_table.iloc[state_o][best_move] += self.l_rate * (reward
                                                                           - self.discount * max(
                            SmartAI.Q_table.iloc[state_n])
                                                                           - SmartAI.Q_table.iloc[state_o][best_move])
                # print(state_o, move, reward)
                return win

            else:
                # exploration
                best_move = random.choice(moves)
                reward, win = self.game.move(self.char, best_move)
                state_n = self.game.get_int_state(self.char)

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


class BrilliantAI(AI):

    train_range: int = 64

    def __init__(self, character: int, state=None):
        super().__init__(character, state)
        self.memo = deque(maxlen=10 ** 5)

        self.MLP_value = DeepQMLP(self.game.size)
        self.MLP_tar = DeepQMLP(self.game.size)
        self.MLP_tar.load_state_dict(self.MLP_value.state_dict())

        self.optimizer = optim.Adam(self.MLP_value.parameters(), lr=self.l_rate * 0.1)

    def move(self, train: bool = False) -> bool:

        if train:
            move = self.get_action() # 1D location
            state_old = self.game.get_array_state(self.char).reshape(self.game.size**2)
            reward, win = self.game.move(self.char, self.game.get_2d_loc(move))
            state_new = self.game.get_array_state(-self.char).reshape(self.game.size**2)
            self.memo.append((state_old, move, reward, win, state_new))

            self.train_long_memory()
            self.learn()
            return win

        else:
            prediction = self.MLP_value(self.game.board)
            move = self.game.get_2d_loc(np.argmax(prediction).item())

            reward, win = self.game.move(self.char, move)
            return win

    def train_long_memory(self) -> torch.tensor:
        sample_range = min(self.train_range, len(self.memo))

        s_o, a_o, r_o, win, s_n = zip(*random.sample(self.memo, sample_range))
        s_o = torch.tensor(np.array(s_o), dtype=torch.float32)
        a_o = torch.tensor(a_o, dtype=torch.long)
        r_o = torch.tensor(r_o, dtype=torch.float32)
        win = torch.tensor(win, dtype=torch.long)
        s_n = torch.tensor(np.array(s_o), dtype=torch.float32)

        values_predict = self.MLP_value(s_o).gather(2, a_o.view(1, -1, 1)).squeeze(2)
        values_predict_next = self.MLP_value(s_n).max(2)[0]
        values_real = r_o - self.discount * values_predict_next + win * self.game.size
        loss = F.mse_loss(values_predict, values_real)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train_short_memory(self):
        pass

    def learn(self):
        self.MLP_tar.load_state_dict(self.MLP_value.state_dict())

    def get_action(self) -> int:

        moves = self.game.get_avail_moves()

        if np.random.uniform() > self.eplis:
            prediction = self.MLP_value(self.game.get_array_state(self.char).reshape(self.game.size**2))    # change the
            action = prediction.argmax().item()
        else:
            action = self.game.get_1d_loc(random.choice(moves))

        return action
