from typing import List, Tuple

import numpy as np
from numpy import ndarray

X = -1
O = 1


class Game:
    size: int
    step: int
    last_player: int
    max_len: int
    board: np.array

    def __init__(self, size=10, win=3):

        assert size >= win

        self.size = size
        self.step = 0
        self.max_len = win
        self.board = np.zeros((size, size), dtype=int)
        self.rewards_X = np.ones((size, size), dtype=float) * -2**self.max_len
        self.rewards_O = np.ones((size, size), dtype=float) * -2**self.max_len
        self.range_min = (int(self.size / 2), int(self.size / 2))
        self.range_max = (int(self.size / 2 + 1), int(self.size / 2 + 1))
        self.is_sim_move = False
        self.update_rewards((5, 5))

    def get_grid_from_state(self, state: int):
        res = []
        for i in range(self.size * self.size):
            res += [state % self.size]
            state = int(state / self.size)

        print(np.array(res).reshape((self.size, self.size)) - 1)

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.step = 0
        self.range_min = (int(self.size / 2), int(self.size / 2))
        self.range_max = (int(self.size / 2 + 1), int(self.size / 2 + 1))
        self.is_sim_move = False

    def update_range(self, loc: tuple):
        min_x = min(max(0, loc[0] - 1), self.range_min[0])
        min_y = min(max(0, loc[1] - 1), self.range_min[1])
        self.range_min = (min_x, min_y)

        max_x = max(min(self.size, loc[0] + 2), self.range_max[0])
        max_y = max(min(self.size, loc[1] + 2), self.range_max[1])
        self.range_max = (max_x, max_y)

    def debug(self) -> None:
        print(self.size, self.step)
        print(self.board)

    def get_avail_moves(self) -> list[tuple]:
        return [tuple(m) for m in np.transpose(np.where(self.board[self.range_min[0]:self.range_max[0],
                                                        self.range_min[1]:self.range_max[1]] == 0)) + self.range_min]

    def get_unavail_moves(self) -> list[tuple]:
        return [tuple(m) for m in np.transpose(np.where(self.board[self.range_min[0]:self.range_max[0],
                                                        self.range_min[1]:self.range_max[1]] != 0)) + self.range_min]

    def check_move(self, move: tuple[int, int]) -> bool:

        if not 0 <= move[0] < self.size:
            return False
        if not 0 <= move[1] < self.size:
            return False

        return self.board[move] == 0

    def move(self, player: int, loc: tuple[int, int]) -> bool:

        if not self.check_move(loc):
            return False

        win = self.check_win_fast(player, loc)

        self.last_player = player
        self.step += 1
        self.board[loc] = player

        if not self.is_sim_move:
            self.update_range(loc)

        self.update_rewards(loc)

        return win

    def sim_move(self, player: int, loc: tuple[int, int]) -> tuple[float, bool, ndarray]:
        self.is_sim_move = True

        reward, win = self.move(player, loc)
        state_new = self.get_array_state_double(player)
        self.board[loc] = 0
        self.step -= 1

        self.is_sim_move = False

        return reward, win, state_new

    def check_win(self, player: int) -> bool:

        row = np.sum(self.board, axis=0)
        for i in row:
            if i == self.size * player:
                return True

        col = np.sum(self.board, axis=1)
        for i in col:
            if i == self.size * player:
                return True

        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == player:
            return True

        if self.board[2, 0] == self.board[1, 1] == self.board[0, 2] == player:
            return True

        return False

    def check_win_fast(self, player: int, loc: tuple[int, int]) -> bool:

        if not 0 <= loc[0] < self.size:
            return False

        if not 0 <= loc[1] < self.size:
            return False

        if not self.board[loc] == 0:
            return False

        bound_min = np.clip([loc[0] - self.max_len, loc[1] - self.max_len], 0, self.size)
        bound_max = np.clip([loc[0] + self.max_len, loc[1] + self.max_len], 0, self.size)

        self.board[loc] = player
        # check y axis
        for i in range(bound_min[1], bound_max[1] - self.max_len + 1):
            if sum(self.board[loc[0], i:i + self.max_len]) == player * self.max_len:
                self.board[loc] = 0
                return True

        # check x axis
        for i in range(bound_min[0], bound_max[0] - self.max_len + 1):
            if sum(self.board[i:i + self.max_len, loc[1]]) == player * self.max_len:
                self.board[loc] = 0
                return True

        # check diagonal
        # l_t -> r_b
        min_dist = np.min(loc - bound_min)
        max_dist = np.min(bound_max - loc)
        for i in range(max_dist + min_dist + 1 - self.max_len):
            s = (self.board[tuple(loc - min_dist + i + j)] for j in range(self.max_len))
            if sum(s) == player * self.max_len:
                self.board[loc] = 0
                return True
        # r_t -> l_b
        # print([bound_min[0], bound_max[1]], [bound_max[0], bound_min[1]])
        min_dist = np.min(np.abs(loc - np.array([bound_max[0] - 1, bound_min[1]])))
        max_dist = np.min(np.abs(np.array([bound_min[0], bound_max[1] - 1]) - loc))
        # print(min_dist, max_dist)
        for i in range(max_dist + min_dist + 2 - self.max_len):
            # print([tuple(loc + np.array([min_dist - i - j, -min_dist + i + j])) for j in range(self.max_len)])
            s = (self.board[tuple(loc + np.array([min_dist - i - j, -min_dist + i + j]))] for j in range(self.max_len))
            # print(s)
            if sum(s) == player * self.max_len:
                self.board[loc] = 0
                return True

        self.board[loc] = 0
        return False

    def get_reward(self, player: int, loc: tuple[int, int]) -> int:

        max_len = self.max_len

        award = 0

        bound_min = np.clip([loc[0] - max_len, loc[1] - max_len], 0, self.size)
        bound_max = np.clip([loc[0] + max_len, loc[1] + max_len], 0, self.size)

        min_dist = np.min(loc - bound_min)
        max_dist = np.min(bound_max - loc)

        min_dist_i = np.min(np.abs(loc - np.array([bound_max[0] - 1, bound_min[1]])))
        max_dist_i = np.min(np.abs(np.array([bound_min[0], bound_max[1] - 1]) - loc))

        exist = self.board[loc] != 0

        if not exist: self.board[loc] = player
        # check y axis

        for len in range(self.max_len, self.max_len - 3, -1):

            is_counted = False

            for i in range(bound_min[1], bound_max[1] - max_len + 1):
                if sum(self.board[loc[0], i:i + max_len]) == player * len:
                    is_counted = True
                    award += 2 ** len
                    break

            # check x axis
            for i in range(bound_min[0], bound_max[0] - max_len + 1):
                if sum(self.board[i:i + max_len, loc[1]]) == player * len:
                    is_counted = True
                    award += 2 ** len
                    break

            # check diagonal
            # l_t -> r_b
            for i in range(max_dist + min_dist + 1 - max_len):
                s = (self.board[tuple(loc - min_dist + i + j)] for j in range(max_len))
                if sum(s) == player * len:
                    is_counted = True
                    award += 2 ** len
                    break

            # r_t -> l_b
            for i in range(max_dist_i + min_dist_i + 2 - max_len):
                # print([tuple(loc + np.array([min_dist - i - j, -min_dist + i + j])) for j in range(self.max_len)])
                s = (self.board[tuple(loc + np.array([min_dist_i - i - j, -min_dist_i + i + j]))] for j in
                     range(max_len))
                # print(s)
                if sum(s) == player * len:
                    is_counted = True
                    award += 2 ** len
                    break

            if is_counted:
                break
        if not exist: self.board[loc] = 0

        return award

    def update_rewards(self, loc):

        if type(loc) != tuple:
            loc = self.get_2d_loc(loc)

        moves = self.get_avail_moves()
        self.rewards_X[loc] = -2 ** self.max_len
        self.rewards_O[loc] = -2 ** self.max_len

        all_loc = []

        bound_min = np.clip([loc[0] - self.max_len+1, loc[1] - self.max_len+1], 0, self.size)
        bound_max = np.clip([loc[0] + self.max_len  , loc[1] + self.max_len]  , 0, self.size)

        # X-Axis direction
        for x in range(bound_min[0], bound_max[0]):
            l = (x, loc[1])
            if l not in moves or l == loc:
                continue
            all_loc += [l]

        # Y-Axis direction
        for y in range(bound_min[1], bound_max[1]):
            l = (loc[0], y)
            if l not in moves or l == loc:
                continue
            all_loc += [l]

        # TopLeft -> BottomRight
        min_dist = np.min(loc - bound_min)
        max_dist = np.min(bound_max - loc)
        ls = [(loc[0]+i, loc[1]+i) for i in range(-min_dist, max_dist)]
        for l in ls:
            if l not in moves or l == loc:
                continue
            all_loc += [l]

        # TopRight -> BottomLet
        min_dist_i = np.min(np.abs(loc - np.array([bound_max[0] - 1, bound_min[1]])))
        max_dist_i = np.min(np.abs(np.array([bound_min[0], bound_max[1] - 1]) - loc))
        ls = [(loc[0]-i, loc[1]+i) for i in range(-min_dist_i, max_dist_i+1)]
        for l in ls:
            if l not in moves or l == loc:
                continue
            all_loc += [l]

        for m in all_loc:
            r_x = self.get_reward(X, m)
            r_o = self.get_reward(O, m)
            dis = self.dist_to_center(m)
            self.rewards_X[m] = r_x + r_o/2 + dis*2
            self.rewards_O[m] = r_o + r_x/2 + dis*2

        for m in moves:
            if self.rewards_X[m] == -1:
                self.rewards_X[m] = self.dist_to_center(m)*2
            if self.rewards_O[m] == -1:
                self.rewards_O[m] = self.dist_to_center(m)*2

    def get_total_reward_at(self, player, loc: tuple):
        if self.board[loc] != 0:
            return 0

        dist = self.dist_to_center(loc)
        return self.get_reward(player, loc) + self.get_reward(-player, loc) / 2 + dist

    def get_reward_grid(self, player: int):
        if player == X:
            return self.rewards_X.reshape(self.size**2) / 2**(self.max_len+1)
        else:
            return self.rewards_O.reshape(self.size**2) / 2**(self.max_len+1)

    def get_win_loc(self, player: int) -> list[tuple]:
        moves = self.get_avail_moves()
        return [m for m in moves if self.check_win_fast(player, m)]

    def get_array_state(self, player: int) -> np.array:
        return self.board * player

    def get_int_state(self, player: int) -> int:
        ret = self.board.reshape(self.size ** 2) * player + 1  # [-1, 1] -> [0, 2]
        loc = 3 ** np.arange(self.size ** 2)
        return int(np.sum(loc * ret))

    def get_array_state_double(self, player: int) -> np.array:
        board = self.board.reshape(self.size ** 2)
        x = np.zeros(self.size ** 2)
        o = np.zeros(self.size ** 2)
        x[np.where(board == X)] = 1
        o[np.where(board == O)] = 1

        if player == X:
            return np.hstack((x, o))
        else:
            return np.hstack((o, x))

    def get_2d_loc(self, loc: int) -> tuple:
        y, x = divmod(loc, self.size)
        return x, y

    def get_1d_loc(self, loc: tuple) -> int:
        return loc[0] + loc[1] * self.size

    def __str__(self) -> str:
        res = ""

        res += "-" * (4 * self.size + 1) + "\n"
        for i in self.board.T:
            res += "|"
            for j in i:
                if j == X:
                    res += " X |"
                elif j == 0:
                    res += "   |"
                elif j == O:
                    res += " O |"
            res += "\n"
            res += "-" * (4 * self.size + 1) + "\n"

        return res

    def display_grid(self) -> None:
        print(self.__str__())

    def count_grid(self):
        x = np.sum(self.board == X)
        o = np.sum(self.board == O)
        n = np.sum(self.board == 0)
        print("X:", x, "O:", o, "N:", 0)

    def __eq__(self, other) -> bool:
        return np.array_equal(self.board, other.board)

    def dist_to_center(self, loc: tuple) -> float:
        if type(loc) != tuple:
            loc = self.get_2d_loc(loc)

        return float(self.size - abs(int(self.size / 2) - loc[0]) - abs(int(self.size / 2) - loc[1]))
