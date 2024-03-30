from stream import Stream
from xf_types import WideType


class Transpose:
    def __init__(
        self,
        t_DataType: type,
        t_ColMemWords: int,
        t_ParEntriesM: int,
        t_ParEntriesN: int,
        p_iterationNum: int,
        p_reuseNum: int,
    ):
        self.t_DataType = t_DataType
        self.t_ColMemWords = t_ColMemWords
        self.t_ParEntriesM = t_ParEntriesM
        self.t_ParEntriesN = t_ParEntriesN

        self.t_BufferSize = t_ParEntriesM * t_ColMemWords

        self.m_iterationNum = p_iterationNum
        self.m_reuseNum = p_reuseNum

    def process(self, p_streamIn: Stream, p_streamOut: Stream):
        p_s0_0 = Stream(WideType)
        p_s0_1 = Stream(WideType)
        p_s1_0 = Stream(WideType)
        p_s1_1 = Stream(WideType)

        l_iter1 = self.m_iterationNum >> 1
        l_iter0 = self.m_iterationNum - l_iter1

        p_s0_0, p_s0_1 = self.split(p_streamIn, p_s0_0, p_s0_1)
        p_s1_0 = self.buffer(l_iter0, p_s0_0, p_s1_0)
        p_s1_1 = self.buffer(l_iter1, p_s0_1, p_s1_1)
        p_streamOut = self.merge(p_s1_0, p_s1_1, p_streamOut)
        return p_streamOut

    def buffer(self, p_iterationNum: int, p_in: Stream, p_out: Stream):
        l_buffer = [
            [
                WideType(t_Width=self.t_ParEntriesN, dtype=self.t_DataType)
                for _ in range(self.t_ColMemWords)
            ]
            for _ in range(self.t_ParEntriesM)
        ]

        for l_block in range(p_iterationNum):
            # read block
            for i in range(self.t_ParEntriesM):
                for j in range(self.t_ColMemWords):
                    l_word = p_in.read()
                    l_buffer[i][j] = l_word

            # stream down l_buffer
            for r in range(self.m_reuseNum):
                for i in range(self.t_ColMemWords):
                    for j in range(self.t_ParEntriesN):
                        l_word = WideType(t_Width=self.t_ParEntriesM, dtype=self.t_DataType)
                        for k in range(self.t_ParEntriesM):
                            l_word[k] = l_buffer[k][i][j]
                        p_out.write(l_word)
        return p_out

    def split(self, p_in: Stream, p_out1: Stream, p_out2: Stream):
        for i in range(self.m_iterationNum):
            for j in range(self.t_BufferSize):
                l_word = p_in.read()
                if (i % 2) == 0:
                    p_out1.write(l_word)
                else:
                    p_out2.write(l_word)
        return p_out1, p_out2

    def merge(self, p_in1: Stream, p_in2: Stream, p_out: Stream):
        for i in range(self.m_iterationNum):
            for r in range(self.m_reuseNum):
                for j in range(self.t_BufferSize):
                    l_word = WideType(t_Width=self.t_ParEntriesM, dtype=self.t_DataType)
                    if (i % 2) == 0:
                        l_word = p_in1.read()
                    else:
                        l_word = p_in2.read()
                    p_out.write(l_word)
        return p_out
