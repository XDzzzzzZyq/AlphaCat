import AlphaCat.Game as Game
from AlphaCat.dumbAI import DumbAI
from AlphaCat.AI import SmartAI

import pandas as pd


def main():
    print("start")

    game = Game.Game(3, 3)
    ai1 = SmartAI(Game.X, game, "q-table")
    ai2 = SmartAI(Game.O, game, "q-table")
    ai3 = DumbAI(Game.O, game)

    test_count = 1000
    debug = test_count < 5
    # testing
    res = {"TestAI": 0, "LabAI": 0, "Draw": 0}
    for i in range(test_count):
        game.reset()
        while True:

            win = ai1.move()
            if debug: game.display_grid()
            if win:
                res["TestAI"] += 1
                break
            if not game.get_avail_moves():
                res["Draw"] += 1
                break

            win = ai3.move()
            if debug: game.display_grid()
            if win:
                res["LabAI"] += 1
                break
            if not game.get_avail_moves():
                res["Draw"] += 1
                break
        if debug: print("-" * 50)
    print("test ", res)


if __name__ == "__main__":
    main()
