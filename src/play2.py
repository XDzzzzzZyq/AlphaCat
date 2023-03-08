import AlphaCat.Game as Game
from AlphaCat.AI import BrilliantAI


def main():
    game = Game.Game(11, 5)
    ai = BrilliantAI(Game.O, game, "Model.pth")
    BrilliantAI.is_train_roles = False
    player = Game.X

    while True:

        move = (-1, -1)
        print(game.get_reward_grid(player).reshape((11, 11)).T*2**(game.max_len+1))
        while not game.check_move(move):
            x = int(input("x: "))
            y = int(input("y: "))
            move = (x, y)
        win = game.check_win_fast(player, move)
        game.move(player, move)
        print(game.board.T)
        #game.display_grid()
        if win:
            print("You Win")
            break
        if not game.get_avail_moves():
            print("Draw")
            break

        win = False
        game.display_grid()
        if win:
            print("AI win")
            break
        if not game.get_avail_moves():
            print("Draw")
            break



if __name__ == "__main__":
    main()
