import AlphaCat.Game as Game
import AlphaCat.dumbAI as ai


def main():
    game = Game.Game(3, 3)

    AI1 = ai.DumbAI(Game.X, game)
    AI2 = ai.DumbAI(Game.O, game)

    round_count = 0;

    while True:

        round_count += 1

        AI1.move()
        if game.check_win(AI1.char):
            print("AI1", round_count)
            game.display_grid()
            break
        if len(game.get_avail_moves()) == 0:
            print("P", round_count)
            game.display_grid()
            break

        AI2.move()
        if game.check_win(AI2.char):
            print("AI2", round_count)
            game.display_grid()
            break
        if len(game.get_avail_moves()) == 0:
            print("P", round_count)
            game.display_grid()
            break


if __name__ == "__main__":
    main()
