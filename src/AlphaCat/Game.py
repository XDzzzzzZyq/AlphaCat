import numpy as np

X = -1
O = 1


class Game:
    size: int
    step: int
    max_len: int
    board: np.array

    def __init__(self, size=10, win=3):

        assert size >= win

        self.size = size
        self.step = 0
        self.max_len = win
        self.board = np.zeros((size, size), dtype=int)

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)

    def debug(self) -> None:
        print(self.size, self.step)
        print(self.board)

    def display_grid(self) -> None:
        res = ""

        res += "-" * (5 * self.size + 1) + "\n"
        for i in self.board.T:
            res += "|"
            for j in i:
                res += f" {j:2} |"
            res += "\n"
            res += "-" * (5 * self.size + 1) + "\n"

        print(res)

    def get_avail_moves(self) -> list[np.array]:

        empty_area = []

        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] == 0:
                    empty_area += [(x, y)]

        return empty_area

    def check_move(self, move: tuple[int, int]) -> bool:

        if not 0 <= move[0] < self.size: raise Exception('move out of boundary')
        if not 0 <= move[1] < self.size: raise Exception('move out of boundary')

        return self.board[move] == 0

    def move(self, player: int, loc: tuple[int, int]):

        if not self.check_move(loc):
            return

        self.board[loc] = player

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
            if np.sum(self.board[loc[0], i:i + self.max_len]) == player * self.max_len:
                self.board[loc] = 0
                return True

        # check x axis
        for i in range(bound_min[0], bound_max[0] - self.max_len + 1):
            if np.sum(self.board[i:i + self.max_len, loc[1]]) == player * self.max_len:
                self.board[loc] = 0
                return True

        # check diagonal
        # l_t -> r_b
        min_dist = np.min(loc - bound_min)
        max_dist = np.min(bound_max - loc)
        for i in range(max_dist + min_dist + 1 - self.max_len):
            s = [self.board[loc - min_dist + i + j] for j in range(self.max_len)]
            if np.sum(s) == player * self.max_len:
                self.board[loc] = 0
                return True
        # r_t -> l_b
        min_dist = np.min(np.abs(loc - np.array([bound_max[0], bound_min[1]])))
        max_dist = np.min(np.abs(np.array([bound_min[0], bound_max[1]]) - loc))
        for i in range(max_dist + min_dist + 1 - self.max_len):
            s = [loc + np.array([bound_max[0] - i - j - 1, bound_min[1] + i + j + 1]) for j in range(self.max_len)]
            if np.sum(s) == player * self.max_len:
                self.board[loc] = 0
                return True

        self.board[loc] = 0
        return False

    def get_win_loc(self, player: int) -> list[list[tuple]]:
        moves = self.get_avail_moves()
        for m in moves:
            if self.check_win_fast(player):
                return m

        return None
