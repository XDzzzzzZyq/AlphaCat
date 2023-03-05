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

    train_count = 5000
    debug = train_count < 5
    res = {"AI1": 0, "AI2": 0, "Draw": 0}
    p = []
    for i in range(1, train_count+1):
        game.reset()
        if i % 10 == 0:
            print("\r    training:", float(i) / train_count, "loss", BrilliantAI.loss, end="")
            p += [[i, BrilliantAI.loss]]
        while True:
            if ai1.rand_move():
                res["AI1"] += 1
                break
            if not game.get_avail_moves():
                res["Draw"] += 1
                break

            if ai2.move(True):
                if debug: game.display_grid()
                res["AI2"] += 1
                break
            if not game.get_avail_moves():
                res["Draw"] += 1
                break

    print(res)
    p = np.array(p).T
    plt.plot(p[0], p[1])
    plt.show()

    train_count = 5000
    res = {"AI1": 0, "AI2": 0, "Draw": 0}
    for i in range(1, train_count+1):
        game.reset()
        while True:
            if ai1.move():
                res["AI1"] += 1
                break
            if not game.get_avail_moves():
                res["Draw"] += 1
                break

            if ai2.move():
                res["AI2"] += 1
                break
            if not game.get_avail_moves():
                res["Draw"] += 1
                break

    print(res)
    game.reset()
    while True:
        if ai1.rand_move():
            game.display_grid()
            break
        if not game.get_avail_moves():
            game.display_grid()
            break
        game.display_grid()
        if ai2.move():
            game.display_grid()
            break
        if not game.get_avail_moves():
            game.display_grid()
            break
        game.display_grid()
    game.count_grid()


if __name__ == "__main__":
    main()
    BrilliantAI.MLP_tar.save()