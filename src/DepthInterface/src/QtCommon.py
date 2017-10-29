import time

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

'''
class Debug:
    def __init__(self):
        self.debug = __debug__

    def print(self, *args, **kwargs):
        print(*args)
        print(**kwargs)
'''

