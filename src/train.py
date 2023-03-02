import AlphaCat.Game as Game
from AlphaCat.dumbAI import DumbAI
from AlphaCat.AI import SmartAI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("start")

    game = Game.Game(3, 3)
    ai1 = SmartAI(Game.X, game, "q-table.xlsx")
    ai2 = SmartAI(Game.O, game, "q-table.xlsx")
    ai3 = DumbAI(Game.O, game)

    train_count = 5000
    test_count = 200

    # training
    print("\n>>>")
    res = {"ai1": 0, "ai2": 0, "draw": 0}
    for i in range(1, int(train_count*3/2+1)):
        if i % 200 == 0: print("\r    training:", float(i) / train_count, end="")
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

    print("\n>>>")
    for i in range(1, int(train_count/2+1)):
        if i % 200 == 0: print("\r    training:", float(i) / train_count, end="")
        game.reset()
        while True:
            if ai3.move():
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

    print("\n>>>")
    for i in range(1, train_count*2+1):
        if i % 200 == 0: print("\r    training:", float(i) / train_count, end="")
        game.reset()
        while True:

            if ai1.move(True):
                res["ai1"] += 1
                break
            if not game.get_avail_moves():
                res["draw"] += 1
                break

            if ai2.move(False):
                res["ai2"] += 1
                break
            if not game.get_avail_moves():
                res["draw"] += 1
                break

    game.reset()
    print("\n>>>")
    print("train", res)

    # testing
    res = {"ai1": 0, "ai2": 0, "draw": 0}
    for i in range(test_count):
        game.reset()
        while True:

            if ai1.rand_move():
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
    SmartAI.Q_table.to_excel("q-table.xlsx")


if __name__ == "__main__":
    main()
