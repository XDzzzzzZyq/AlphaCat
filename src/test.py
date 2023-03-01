import AlphaCat.Game as Game
import AlphaCat.dumbAI as aid
import AlphaCat.AI as ais

def main():
    print("start")

    game = Game.Game(3, 3)
    ai1 = ais.SmartAI(Game.X, game, "q-table.xlsx")
    ai2 = ais.SmartAI(Game.O, game, "q-table.xlsx")
    ai3 = aid.DumbAI(Game.O, game)

    test_count = 2
    debug = test_count < 5
    # testing
    res = {"ai1": 0, "ai2": 0, "draw": 0}
    for i in range(test_count):
        game.reset()
        while True:

            win = ai1.rand_move()
            if debug: game.display_grid()
            if win:
                res["ai1"] += 1
                break
            if not game.get_avail_moves():
                res["draw"] += 1
                break

            win = ai2.move()
            if debug: game.display_grid()
            if win:
                res["ai2"] += 1
                break
            if not game.get_avail_moves():
                res["draw"] += 1
                break
        if debug: print("-"*50)
    print("test ", res)
    ais.SmartAI.Q_table.to_excel("q-table.xlsx")


if __name__ == "__main__":
    main()
