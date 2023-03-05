import numpy as np
import pandas as pd

import AlphaCat.Game as Game
import torch
from collections import deque

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

    # memo = deque(maxlen=3)
    #memo.append((1, 2, "asd", False))
    #memo.append((21, 2, "a ssd", False))
    #memo.append((1, 2, "assd", True))

    #print(memo)

    a = [[[1, 2, 3], [4, 5, 6]]]
    a = torch.tensor(a)
    print(a)

    index = [[[1], [2]]]
    index = torch.tensor(index)
    print(a.gather(2, index))
    print(a!=1)

    a = torch.rand(1, 7)
    print(a)
    a = a.gather(0, torch.tensor([[2,3,5]]))
    print(a)
    print(a.argmax().item())

    print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability())


if __name__ == "__main__":
    main()
