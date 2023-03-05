import numpy as np
import pandas as pd

import AlphaCat.Game as Game


def main():
    board = Game.Game(15, 5)
    board.debug()
    board.move(Game.X, (10, 8))
    board.move(Game.X, (11, 9))
    board.move(Game.O, (12, 10))
    board.display_grid()
    print(board.get_award(Game.X, (9, 7)))
    # print(board.get_award(Game.O, (0, 0)))
    # board.get_grid_from_state(2638)
    # print("run")


if __name__ == "__main__":
    main()
