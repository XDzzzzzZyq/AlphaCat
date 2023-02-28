import sys
import os

def f():
    print(__file__)

if "workspaces" in __file__:
    sys.path.append("/workspaces/AlphaCat/src/AlphaCat")                            # for Github Codespace
else:
    sys.path.append(os.path.dirname(os.path.dirname(__file__))+"/src/AlphaCat")     # for local debug

import Game as Game
import dumbAI as DAI
