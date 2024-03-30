from stream import Stream
from xf_types import WideType


# Implementation of Matrix Buffer class with true, false arguments for the last two template parameters
class MatrixBuffer:
    def __init__(
        self,
        t_DataType: type,
        t_MemWidth: int,
        t_bkDim: int,
        t_bColMemWords: int,
    ):
        self.t_DataType = t_DataType
        self.t_MemWidth = t_MemWidth
        self.t_bkDim = t_bkDim
        self.t_bColMemWords = t_bColMemWords
        self.t_BufferSize = t_bkDim * t_bColMemWords

    def process(
        self,
        p_streamIn: Stream,
        p_streamOut: Stream,
        p_iterationNum: int,
        p_reuseNum: int = 1,
    ) -> Stream:
        p_s0_0, p_s0_1, p_s1_0, p_s1_1 = (
            Stream(dtype=WideType),
            Stream(dtype=WideType),
            Stream(dtype=WideType),
            Stream(dtype=WideType)
        )

        p_s0_0, p_s0_1 = self.split(p_iterationNum, p_streamIn, p_s0_0, p_s0_1)
        p_s1_0 = self.buffer(
            (p_iterationNum // 2) + (p_iterationNum % 2), p_s0_0, p_s1_0, p_reuseNum
        )
        p_s1_1 = self.buffer(p_iterationNum // 2, p_s0_1, p_s1_1, p_reuseNum)
        p_streamOut = self.merge(
            p_iterationNum, p_s1_0, p_s1_1, p_streamOut, p_reuseNum
        )
        return p_streamOut

    def buffer(
        self, p_iterationNum: int, p_in: Stream, p_out: Stream, p_reuseNum: int
    ) -> Stream:
        l_buffer = [
            [
                WideType(self.t_MemWidth, dtype=self.t_DataType)
                for _ in range(self.t_bColMemWords)
            ]
            for _ in range(self.t_bkDim)
        ]

        for l_block in range(p_iterationNum):
            # read block
            for i in range(self.t_bkDim):
                for j in range(self.t_bColMemWords):
                    l_word = p_in.read()
                    l_buffer[i][j] = l_word

            # stream down l_buffer
            for i in range(p_reuseNum):
                for k in range(self.t_bColMemWords):
                    for l in range(self.t_bkDim):
                        l_word = l_buffer[l][k]
                        p_out.write(l_word)
        return p_out

    def split(self, p_iterationNum: int, p_in: Stream, p_out1: Stream, p_out2: Stream):
        for i in range(p_iterationNum):
            for j in range(self.t_BufferSize):
                l_word = p_in.read()
                if (i % 2) == 0:
                    p_out1.write(l_word)
                else:
                    p_out2.write(l_word)
        return p_out1, p_out2

    def merge(
        self,
        p_iterationNum,
        p_in1: Stream,
        p_in2: Stream,
        p_out: Stream,
        p_reuseNum: int,
    ):
        for i in range(p_iterationNum):
            for r in range(p_reuseNum):
                for j in range(self.t_BufferSize):
                    l_word = WideType(t_Width=self.t_MemWidth, dtype=self.t_DataType)
                    if (i % 2) == 0:
                        l_word = p_in1.read()
                    else:
                        l_word = p_in2.read()
                    p_out.write(l_word)
        return p_out
