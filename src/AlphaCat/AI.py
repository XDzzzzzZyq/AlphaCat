import Game
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from collections import deque

from MLP import DeepQMLP


class AI:

    eplis = 0.2
    l_rate = 0.1
    discount = 0.5

    def __init__(self, character: int, state=None):
        self.game = state
        self.char = character

    def move(self, train: bool = False) -> bool:
        pass

    def get_action(self) -> tuple:
        pass

    def rand_move(self):
        moves = self.game.get_avail_moves()
        if moves is not None:
            dist = 2 ** np.array([self.game.dist_to_center(m) for m in moves])
            dist /= sum(dist)
            moves = [self.game.get_1d_loc(m) for m in moves]
            move = np.random.choice(moves, p=dist)
            win = self.game.move(self.char, self.game.get_2d_loc(move))
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
            _, action = max((self.Q_table[self.game.board][act], act) for act in moves)
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
                reward = self.game.get_reward_grid(self.char)[best_move]
                win = self.game.move(self.char, self.game.get_2d_loc(best_move))

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
                reward = self.game.get_reward_grid(self.char)[best_move]
                win = self.game.move(self.char, best_move)
                state_n = self.game.get_int_state(self.char)

                SmartAI.Q_table.iloc[state_o][self.game.get_1d_loc(best_move)] += self.l_rate * (
                        reward
                        - self.discount * max(SmartAI.Q_table.iloc[state_n])
                        - SmartAI.Q_table.iloc[state_o][self.game.get_1d_loc(best_move)])

                return win

        else:

            max_v = np.max(SmartAI.Q_table.iloc[state_o][avail])
            best_move = np.random.choice([i for i in avail if SmartAI.Q_table.iloc[state_o][i] == max_v])
            win = self.game.move(self.char, self.game.get_2d_loc(best_move))
            # self.game.display_grid()
            return win


class BrilliantAI(AI):

    train_range: int = 128
    loss = None
    is_train_roles = True
    memo = deque(maxlen=10 ** 3)
    game_size = 0

    MLP_reward = MLP_value = MLP_tar = None
    optimizer_reward = optimizer = None

    def __init__(self, character: int, state=None, Model=None):
        super().__init__(character, state)

        BrilliantAI.game_size = state.size

        if not BrilliantAI.MLP_value:
            if not Model:
                BrilliantAI.MLP_reward = DeepQMLP(self.game.size)
                BrilliantAI.MLP_value = DeepQMLP(self.game.size)
                BrilliantAI.MLP_tar = DeepQMLP(self.game.size)
                BrilliantAI.MLP_tar.load_state_dict(BrilliantAI.MLP_value.state_dict())
                BrilliantAI.optimizer_reward = optim.Adam(BrilliantAI.MLP_reward.parameters(), lr=0.1)
                BrilliantAI.optimizer = optim.Adam(BrilliantAI.MLP_value.parameters(), lr=0.1)
            else:
                Model = os.path.dirname(__file__) + "\\" + Model
                BrilliantAI.MLP_value = DeepQMLP(self.game.size)
                BrilliantAI.MLP_tar = DeepQMLP(self.game.size)
                state_dict = torch.load(Model)
                BrilliantAI.MLP_value.load_state_dict(state_dict)
                BrilliantAI.MLP_tar.load_state_dict(state_dict)
                BrilliantAI.optimizer = optim.Adam(BrilliantAI.MLP_value.parameters(), lr=0.1)

    def move(self, train: bool = False) -> bool:

        move = self.get_action(not train)  # 1D location

        if train:
            if BrilliantAI.is_train_roles:

                state_old = self.game.get_array_state_double(self.char)
                reward = self.game.get_reward_grid(self.char)
                win = self.game.move(self.char, self.game.get_2d_loc(move))
                BrilliantAI.memo.append((state_old, None, reward, None, None))

                return win
            else:

                # state_old = self.game.get_array_state(self.char).reshape(self.game.size**2)
                state_old = self.game.get_array_state_double(self.char)
                reward = self.game.get_reward_grid(self.char)
                win = self.game.move(self.char, self.game.get_2d_loc(move))
                # state_new = self.game.get_array_state(-self.char).reshape(self.game.size**2)
                state_new = self.game.get_array_state_double(-self.char)
                BrilliantAI.memo.append((state_old, move, reward, win, state_new))

                return win

        else:
            win = self.game.move(self.char, self.game.get_2d_loc(move))
            return win

    @classmethod
    def train_long_memory(cls) -> float:
        if len(cls.memo) < cls.train_range:
            return -1

        s_o, a_o, r_o, win, s_n = zip(*random.sample(cls.memo, cls.train_range))
        s_o = torch.tensor(np.array(s_o), dtype=torch.float32)  # [128, 242]
        a_o = torch.tensor(a_o, dtype=torch.long)               # [128, 1  ]
        r_o = torch.tensor(r_o, dtype=torch.float32)            # [128, 121]
        win = torch.tensor(win, dtype=torch.long)               # [128, 1  ]
        s_n = torch.tensor(np.array(s_n), dtype=torch.float32)  # [128, 242]

        values_predict = BrilliantAI.MLP_value(s_o).gather(0, a_o.view(-1, 1))
        values_predict_next = BrilliantAI.MLP_value(s_n).max(1)[0]
        values_real = r_o.gather(0, a_o.view(-1, 1)) - (1-win) * cls.discount * values_predict_next + win * cls.game_size
        loss = F.mse_loss(values_predict.squeeze(), values_real)

        BrilliantAI.optimizer.zero_grad()
        loss.backward()
        BrilliantAI.optimizer.step()
        return loss.item()

    # reward is the role
    @classmethod
    def train_roles(cls):
        if len(BrilliantAI.memo) < cls.train_range:
            return -1

        s_o, _, r_o, _, _ = zip(*random.sample(BrilliantAI.memo, cls.train_range))
        s_o = torch.tensor(np.array(s_o), dtype=torch.float32)  # [128, 242]
        r_o = torch.tensor(np.array(r_o), dtype=torch.float32)  # [128, 121]

        values_predict = BrilliantAI.MLP_reward(s_o)
        loss = F.mse_loss(values_predict.squeeze(), r_o)

        BrilliantAI.optimizer_reward.zero_grad()
        loss.backward()
        BrilliantAI.optimizer_reward.step()
        return loss.item()

    def get_action(self, no_rand=False) -> int:

        moves = [self.game.get_1d_loc(m) for m in self.game.get_avail_moves()]
        un_moves = [self.game.get_1d_loc(m) for m in self.game.get_unavail_moves()]

        if np.random.uniform() > self.eplis or no_rand:
            # state = self.game.get_array_state(self.char).reshape(self.game.size**2)
            state = self.game.get_array_state_double(self.char)

            if BrilliantAI.is_train_roles:
                prediction = BrilliantAI.MLP_reward(state)
            else:
                prediction = BrilliantAI.MLP_value(state)

            _, action = max((prediction[0, a], a) for a in moves)
        else:
            dist = 2 ** np.array([self.game.dist_to_center(m) for m in moves])
            dist /= sum(dist)
            action = np.random.choice(moves, p=dist)
        return action

    @classmethod
    def learn(cls):
        if cls.is_train_roles:
            cls.MLP_value.load_state_dict(cls.MLP_reward.state_dict())
        else:
            cls.MLP_tar.load_state_dict(cls.MLP_value.state_dict())

    @classmethod
    def finish_train_role(cls):
        cls.learn()
        cls.is_train_roles = False
        cls.learn()
        cls.memo = deque(maxlen=10 ** 3)
