import AlphaCat.Game as Game
import AlphaCat.dumbAI as aid
import AlphaCat.AI as ais

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("start")

    game = Game.Game(3, 3)
    ai1 = ais.SmartAI(Game.X, game)
    ai2 = ais.SmartAI(Game.O, game)
    ai3 = aid.DumbAI(Game.O, game)

    train_count = 10000
    test_count = 200

    # training
    res = {"ai1": 0, "ai2": 0, "draw": 0}
    for i in range(train_count):
        if i % 100 == 0: print("    training:", float(i) / train_count)
        game.reset()
        while True:

            if ai1.rand_move():
                res["ai1"] += 1
                break
            if not game.get_avail_moves():
                res["draw"] += 1
                break

            if ai2.move(True):
                res["ai2"] += 1
                break
            if not game.get_avail_moves():
                res["draw"] += 1
                break
        # game.display_grid()
    game.reset()
    print("train", res)
    # testing
    res = {"ai1": 0, "ai2": 0, "draw": 0}
    for i in range(test_count):
        game.reset()
        while True:

            if ai1.move():
                res["ai1"] += 1
                break
            if not game.get_avail_moves():
                res["draw"] += 1
                break

            if ai2.move():
                res["ai2"] += 1
                break
            if not game.get_avail_moves():
                res["draw"] += 1
                break

    print("test ", res)
    ais.SmartAI.Q_table.to_excel("q-table.xlsx")


if __name__ == "__main__":
    main()
