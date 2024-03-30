from xf_types import WideType, MemIntType
from stream import Stream


class WideTypeBuffer2D:
    # Size in number of MemWords in each dimension
    # self.size[0] = number of memwords in row dim
    # self.size[1] = number of memwords in col dim
    def __init__(self, t_MemWidth: int, size: tuple):
        self.t_MemWidth = t_MemWidth
        self.size = size
        self.buffer = [
            [WideType(t_Width=t_MemWidth) for i in range(size[1])]
            for j in range(t_MemWidth * size[0])
        ]

    def Clear(self):
        for i in range(self.t_MemWidth * self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.t_MemWidth):
                    self.buffer[i][j][k] = 0

    def BufferBlock(self, p_stream: Stream):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for l in range(self.t_MemWidth):
                    l_val = p_stream.read()
                    for k in range(self.t_MemWidth):
                        self.buffer[i * self.t_MemWidth + l][j][k] += l_val[k]

    def DownStream(self, p_stream: Stream):
        for i in range(self.t_MemWidth * self.size[0]):
            for j in range(self.size[1]):
                p_stream.write(self.buffer[i][j])
        return p_stream

    def BufferBlockStrassens(
        self, p_stream: Stream, out_indices: tuple, add_or_sub: tuple
    ):
        # size of buffer is 2x2 blocks, each of size t_MemWidth * size[0] x size[1]
        # the incoming block is to be placed in each of out_indices, where out_indices
        # is a tuple of tuples, each tuple being a pair of indices (row, col)
        # add_or_sub is a tuple containing boolean indicating whether to add or subtract
        for i in range(self.size[0] // 2):
            for j in range(self.size[1] // 2):
                for l in range(self.t_MemWidth):
                    l_val = p_stream.read()
                    for k in range(self.t_MemWidth):
                        for block_idx, (block_r, block_c) in enumerate(out_indices):
                            if add_or_sub[block_idx]:
                                self.buffer[
                                    (i + block_r * self.size[0] // 2) * self.t_MemWidth
                                    + l
                                ][(j + block_c * self.size[1] // 2)][k] += l_val[k]
                            else:
                                self.buffer[
                                    (i + block_r * self.size[0] // 2) * self.t_MemWidth
                                    + l
                                ][(j + block_c * self.size[1] // 2)][k] -= l_val[k]

    def DownStreamStrassen(self, p_stream: Stream):
        for i in range(self.t_MemWidth * self.size[0]):
            for j in range(self.size[1]):
                p_stream.write(self.buffer[i][j])
        return p_stream


class GemmCBuffer:
    def __init__(
        self,
        t_MemWidth: int,
        t_aRowMemWords: int,
        t_bColMemWords: int,
        p_aColBlocks: int,
        p_cBlocks: int,
    ):
        self.t_MemWidth = t_MemWidth
        self.t_aRowMemWords = t_aRowMemWords
        self.t_bColMemWords = t_bColMemWords
        self.p_aColBlocks = p_aColBlocks
        self.p_cBlocks = p_cBlocks

        self.buffer = WideTypeBuffer2D(t_MemWidth, (t_aRowMemWords, t_bColMemWords))

    def Clear(self):
        self.buffer.Clear()

    def BufferAndDownStream(self, p_Cs):
        for l_block in range(self.p_cBlocks):
            self.Clear()
            for m in range(self.p_aColBlocks):
                self.buffer.BufferBlock(p_Cs)
            p_Cs = self.buffer.DownStream(p_Cs)
        return p_Cs


class StrassensCBuffer:
    def __init__(
        self,
        t_MemWidth: int,
        t_aRowMemWords: int,
        t_bColMemWords: int,
        p_aColBlocks: int,
        p_cBlocks: int
    ):
        self.t_MemWidth = t_MemWidth
        self.t_aRowMemWords = t_aRowMemWords
        self.t_bColMemWords = t_bColMemWords
        self.p_aColBlocks = p_aColBlocks
        self.p_cBlocks = p_cBlocks

        self.buffer = WideTypeBuffer2D(
            t_MemWidth, (2 * t_aRowMemWords, 2 * t_bColMemWords)
        )

    def Clear(self):
        self.buffer.Clear()

    def BufferAndDownStream(self, p_Cs: Stream) -> Stream:
        p_Outs = Stream(dtype=WideType)
        for l_block in range(self.p_cBlocks):
            self.Clear()
            for m in range(self.p_aColBlocks):
                self.buffer.BufferBlockStrassens(p_Cs, ((0, 0), (1, 1)), (True, True))  # M1
                self.buffer.BufferBlockStrassens(p_Cs, ((1, 0), (1, 1)), (True, False))  # M2
                self.buffer.BufferBlockStrassens(p_Cs, ((0, 1), (1, 1)), (True, True))  # M3
                self.buffer.BufferBlockStrassens(p_Cs, ((0, 0), (1, 0)), (True, True))  # M4
                self.buffer.BufferBlockStrassens(p_Cs, ((0, 0), (0, 1)), (False, True))  # M5
                self.buffer.BufferBlockStrassens(p_Cs, ((1, 1),), (True,))  # M6
                self.buffer.BufferBlockStrassens(p_Cs, ((0, 0),), (True,))  # M7
            p_Outs = self.buffer.DownStreamStrassen(p_Outs)
        return p_Outs
