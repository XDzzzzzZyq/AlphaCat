import AlphaCat.Game as Game
from AlphaCat.AI import BrilliantAI
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


def main():
    print("start")

    summary = SummaryWriter(log_dir="training_data")

    game = Game.Game(11, 5)
    ai1 = BrilliantAI(Game.X, game)
    ai2 = BrilliantAI(Game.O, game)

    train_count = 2000
    debug = train_count < 5
    res = {"AI1": 0, "AI2": 0, "Draw": 0}
    p = []
    t = time.time()
    for i in range(1, train_count+1):
        game.reset()
        if i % 50 == 0:
            BrilliantAI.loss = BrilliantAI.train_roles()
            print(f"\r    training: {float(i) / train_count:.1%} loss: {BrilliantAI.loss:.2f} time: {time.time() - t:.1f}s", end="")
            summary.add_scalar(tag="roles/loss", scalar_value=BrilliantAI.loss, global_step=i)
            summary.add_scalar(tag="roles/time", scalar_value=time.time() - t, global_step=i)
            t = time.time()
            BrilliantAI.learn()
            p += [[i, BrilliantAI.loss]]
        while True:
            if ai1.move(True):
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

    print("\n", res)
    game.reset()
    BrilliantAI.finish_train_role()

    train_count = 5000*0
    res = {"AI1": 0, "AI2": 0, "Draw": 0}
    for i in range(1, train_count+1):
        game.reset()
        if i % 50 == 0:
            BrilliantAI.loss = BrilliantAI.train()
            print(f"\r    training: {float(i) / train_count:.1%} loss: {BrilliantAI.loss:.2f} time: {time.time() - t:.1f}s", end="")
            summary.add_scalar(tag="train/loss", scalar_value=BrilliantAI.loss, global_step=i + 2000)
            summary.add_scalar(tag="train/time", scalar_value=time.time() - t, global_step=i + 2000)
            t = time.time()
            BrilliantAI.learn()
            p += [[i + 2000, BrilliantAI.loss]]
        while True:
            if ai1.move(True):
                res["AI1"] += 1
                break
            if not game.get_avail_moves():
                res["Draw"] += 1
                break

            if ai2.move(True):
                res["AI2"] += 1
                break
            if not game.get_avail_moves():
                res["Draw"] += 1
                break

    print("\n", res)
    p = np.array(p).T
    plt.plot(p[0], p[1])
    plt.show()
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