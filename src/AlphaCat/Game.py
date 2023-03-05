from typing import List

import numpy as np

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

    def get_grid_from_state(self, state: int):
        res = []
        for i in range(self.size * self.size):
            res += [state % self.size]
            state = int(state / self.size)

        print(np.array(res).reshape((self.size, self.size)) - 1)

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.step = 0

    def debug(self) -> None:
        print(self.size, self.step)
        print(self.board)

    def get_avail_moves(self) -> list[tuple]:
        return [tuple(m) for m in np.transpose(np.where(self.board == 0))]

    def check_move(self, move: tuple[int, int]) -> bool:

        if not 0 <= move[0] < self.size:
            return False
        if not 0 <= move[1] < self.size:
            return False

        return self.board[move] == 0

    def move(self, player: int, loc: tuple[int, int]) -> tuple[int, bool]:

        if not self.check_move(loc):
            return -self.max_len, False

        win = self.check_win_fast(player, loc)

        if self.max_len < 5:
            reward = int(win) * 3
            reward_e = self.check_win_fast(-player, loc)
        else:
            reward = self.get_award(player, loc)
            reward_e = self.get_award(-player, loc) / 2

        self.last_player = player
        self.step += 1
        self.board[loc] = player

        if self.max_len < 5:
            moves_n = self.get_avail_moves()
            for m in moves_n:
                if self.check_win_fast(-player, m):
                    reward -= 1

        return reward + reward_e, win

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

    def get_award(self, player: int, loc: tuple[int, int]) -> int:

        max_len = self.max_len

        award = 0

        bound_min = np.clip([loc[0] - max_len, loc[1] - max_len], 0, self.size)
        bound_max = np.clip([loc[0] + max_len, loc[1] + max_len], 0, self.size)

        min_dist = np.min(loc - bound_min)
        max_dist = np.min(bound_max - loc)

        min_dist_i = np.min(np.abs(loc - np.array([bound_max[0] - 1, bound_min[1]])))
        max_dist_i = np.min(np.abs(np.array([bound_min[0], bound_max[1] - 1]) - loc))

        self.board[loc] = player
        # check y axis

        for len in range(self.max_len, self.max_len - 3, -1):

            is_counted = False

            for i in range(bound_min[1], bound_max[1] - max_len + 1):
                if sum(self.board[loc[0], i:i + max_len]) == player * len:
                    is_counted = True
                    award += 2 ** len

            # check x axis
            for i in range(bound_min[0], bound_max[0] - max_len + 1):
                if sum(self.board[i:i + max_len, loc[1]]) == player * len:
                    is_counted = True
                    award += 2 ** len

            # check diagonal
            # l_t -> r_b
            for i in range(max_dist + min_dist + 1 - max_len):
                s = (self.board[tuple(loc - min_dist + i + j)] for j in range(max_len))
                if sum(s) == player * len:
                    is_counted = True
                    award += 2 ** len

            # r_t -> l_b
            for i in range(max_dist_i + min_dist_i + 2 - max_len):
                # print([tuple(loc + np.array([min_dist - i - j, -min_dist + i + j])) for j in range(self.max_len)])
                s = (self.board[tuple(loc + np.array([min_dist_i - i - j, -min_dist_i + i + j]))] for j in
                     range(max_len))
                # print(s)
                if sum(s) == player * len:
                    is_counted = True
                    award += 2 ** len

            if is_counted:
                break

        self.board[loc] = 0

        return award

    def get_win_loc(self, player: int) -> list[tuple]:
        moves = self.get_avail_moves()
        return [m for m in moves if self.check_win_fast(player, m)]

    def get_array_state(self, player: int) -> np.array:
        return self.board * player

    def get_int_state(self, player: int) -> int:
        ret = self.board.reshape(self.size ** 2) * player + 1  # [-1, 1] -> [0, 2]
        loc = 3 ** np.arange(self.size ** 2)
        return int(np.sum(loc * ret))

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

    def __eq__(self, other) -> bool:
        return np.array_equal(self.board, other.board)
