import AlphaCat.Game as Game
from AlphaCat.dumbAI import DumbAI
from AlphaCat.AI import SmartAI


def main():
    game = Game.Game(3, 3)
    ai = SmartAI(Game.X, game, "q-table")
    player = Game.O

    while True:

        win = ai.move()
        game.display_grid()
        if win:
            print("AI win")
            break
        if not game.get_avail_moves():
            print("Draw")
            break

        move = (-1, -1)
        while not game.check_move(move):
            move = game.get_2d_loc(int(input("move: ")))
        win = game.check_win_fast(player, move)
        game.move(player, move)
        game.display_grid()
        if win:
            print("You Win")
            break
        if not game.get_avail_moves():
            print("Draw")
            break


if __name__ == "__main__":
    main()
