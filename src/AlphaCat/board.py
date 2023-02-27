
class Board:


    def __init__(self, _size = 10):
        self.size = _size
        self.step = 0

    def debug(self):
        print(self.size, self.step)