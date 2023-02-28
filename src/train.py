import AlphaCat.Game as Game
import AlphaCat.dumbAI as aid
import AlphaCat.AI as ais

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("start")

    train_count = 0

    game = Game.Game(3, 3)
    ai1 = ais.SmartAI(Game.X, game)
    ai2 = ais.SmartAI(Game.O, game)
    ai3 = aid.DumbAI(Game.O, game)

    # training
    res = {"ai1": 0, "ai2": 0, "draw": 0}
    while train_count < 50:
        game.reset()
        while True:

            if ai1.move():
                res["ai1"] += 1
                break
            if not game.get_avail_moves():
                res["draw"] += 1
                break

            if ai3.move():
                res["ai2"] += 1
                break
            if not game.get_avail_moves():
                res["draw"] += 1
                break

        train_count += 1
        # game.display_grid()
    game.reset()
    print(res)
    # testing
    res = {"ai1": 0, "ai2": 0, "draw": 0}
    test_count = 0
    while test_count < 0:
        game.reset()
        while True:

            if ai1.move():
                res["ai1"] += 1
                break
            if not game.get_avail_moves():
                res["draw"] += 1
                break

            if ai3.move():
                res["ai2"] += 1
                break
            if not game.get_avail_moves():
                res["draw"] += 1
                break

        test_count += 1

    print(res)


if __name__ == "__main__":
    main()
