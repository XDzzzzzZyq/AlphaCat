import numpy as np
import pandas as pd

import AlphaCat.Game as Game


def main():
    arr = np.arange(10)
    rd = np.random.randint(1, 10, 5)

    print(arr)
    print(rd)

    s = pd.Series(rd)
    data = pd.DataFrame({"s": s, "w": s})
    print(data)

    board = Game.Game(3, 3)
    board.debug()
    board.display_grid()
    print(board.get_avail_moves())
    board.move(Game.X, (0, 2))
    board.move(Game.X, (0, 1))
    print(board.get_avail_moves())
    board.display_grid()
    #print(board.check_win_fast(Game.X, (2, 2)))
    print(board.get_award(Game.O, (0, 0)))
    #board.get_grid_from_state(2638)
    #print("run")

if __name__ == "__main__":
    main()
