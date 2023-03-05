import AlphaCat.Game as Game
from AlphaCat.AI import BrilliantAI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


game = Game.Game(3, 3)
ai1 = BrilliantAI(Game.X, game)
ai2 = BrilliantAI(Game.O, game)

while True:

    win = ai1.move()
    if win:
        break
    if not game.get_avail_moves():
        break

    win = ai2.move()
    if win:
        break
    if not game.get_avail_moves():
        break

