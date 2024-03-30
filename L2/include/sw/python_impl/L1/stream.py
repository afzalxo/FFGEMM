
class Stream:
    def __init__(self, dtype=int, depth=1):
        self.dtype = dtype
        self.depth = depth
        # Create fifo buffer
        self.buffer = []

    def write(self, data):
        self.buffer.append(self.dtype(data))

    def read(self):
        if len(self.buffer) == 0:
            return None
        return self.buffer.pop(len(self.buffer) - 1)
