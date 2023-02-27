import numpy as np
import pandas as pd

import AlphaCat.board as Aboard

def main():
    arr = np.arange(10)
    rd = np.random.randint(1, 10, 5)

    print(arr)
    print(rd)

    s = pd.Series(rd)
    data = pd.DataFrame({"s":s, "w":s})
    print(data)

    board = Aboard.Board(8)
    board.debug();



if __name__ == "__main__":
    main()