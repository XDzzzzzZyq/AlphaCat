import AlphaCat.Game as Game
from AlphaCat.AI import BrilliantAI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("start")

    game = Game.Game(11, 5)
    ai1 = BrilliantAI(Game.X, game)
    ai2 = BrilliantAI(Game.O, game)

    train_count = 1000
    debug = train_count < 5
    for i in range(train_count):
        game.reset()
        if i % 10 == 0: print("\r    training:", float(i) / train_count, "loss", 1, end="")
        while True:
            if ai1.rand_move():
                if debug: game.display_grid()
                break
            if not game.get_avail_moves():
                if debug: game.display_grid()
                break
            if debug: game.display_grid()
            if ai2.move(True):
                if debug: game.display_grid()
                break
            if not game.get_avail_moves():
                if debug: game.display_grid()
                break
            if debug: game.display_grid()


if __name__ == "__main__":
    main()