import AlphaCat.Game as Game
import AlphaCat.dumbAI as ai

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():

    print("start")
    res = {"ai1": 0, "ai2": 0, "draw": 0}

    round_count = 0;

    game = Game.Game(3, 3)
    ai1 = ai.DumbAI(Game.X, game)
    ai2 = ai.DumbAI(Game.O, game)
    while round_count < 50:
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

        round_count += 1
        #game.display_grid()

    print(res)


if __name__ == "__main__":
    main()
