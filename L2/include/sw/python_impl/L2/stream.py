from matrix import Matrix
from xf_types import WideType
import copy


class Stream:
    def __init__(self, dtype=int, depth=1):
        self.dtype = dtype
        self.depth = depth
        # Create fifo buffer
        self.buffer = []

    def write(self, data):
        assert isinstance(data, self.dtype), "Data type mismatch"
        self.buffer.append(copy.deepcopy(data))

    def read(self):
        if len(self.buffer) == 0:
            return None
        return self.buffer.pop(0)

    def split(self, numMemWords: int):
        assert numMemWords > 0, "numMemWords must be greater than 0"
        assert len(self.buffer) >= numMemWords, "Not enough data in stream"
        new_stream = Stream(self.dtype, self.depth)
        for i in range(numMemWords):
            new_stream.write(self.read())
        return new_stream

    def __str__(self):
        return str(self.buffer)

    def __repr__(self):
        return repr(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def append(self, other):
        assert isinstance(other, Stream), "Data type mismatch"
        self.buffer.extend(other.buffer)

    def from_matrix(self, dtype: type, t_MemWidth: int, matrix: Matrix):
        matrix_as_list = matrix.as_list()
        for i in range(0, len(matrix_as_list), t_MemWidth):
            memword = WideType(
                t_MemWidth, dtype=dtype
            )
            for j in range(t_MemWidth):
                memword[j] = matrix_as_list[i + j]
            self.write(memword)
        return self
