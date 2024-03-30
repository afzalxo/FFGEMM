from xf_types import WindowRm, TriangSrl, WideType, TaggedFloat
from matrix import Matrix
from stream import Stream


class Gemm:
    def __init__(self,
                 t_Datatype: type,
                 t_KBufferDim: int,
                 t_ParEntriesM: int,
                 t_ParEntriesN: int):
        self.t_ParEntriesM = t_ParEntriesM
        self.t_ParEntriesN = t_ParEntriesN
        self.t_ParEntries = self.t_ParEntriesM
        self.t_DataType = t_Datatype
        self.t_KBufferDim = t_KBufferDim
        assert (t_KBufferDim >= t_ParEntriesM + t_ParEntriesN)

    def gemm(self, p_As: Stream, p_Bs: Stream, p_blocks: int) -> Stream:
        l_awin = WindowRm(self.t_ParEntries, self.t_ParEntries, dtype=TaggedFloat)
        l_bwin = WindowRm(self.t_ParEntries, self.t_ParEntries, dtype=TaggedFloat)
        l_Ta = TriangSrl(self.t_ParEntries, dtype=TaggedFloat)
        l_Tb = TriangSrl(self.t_ParEntries, dtype=TaggedFloat)

        l_C = [WideType(self.t_ParEntries, dtype=self.t_DataType) for _ in range(self.t_ParEntries)]
        l_Co = [WideType(self.t_ParEntries, dtype=self.t_DataType) for _ in range(self.t_ParEntries)]

        # Output stream
        p_sum = Stream(dtype=WideType)

        for l in range(p_blocks+1):
            for k in range(self.t_KBufferDim):
                l_A = WideType(self.t_ParEntries, dtype=int)
                l_B = WideType(self.t_ParEntries, dtype=int)
                # print(l_A)

                if l < p_blocks:
                    l_A = p_As.read()
                    l_B = p_Bs.read()

                # print(l_A)

                l_avec = WideType(self.t_ParEntries, dtype=TaggedFloat)
                l_bvec = WideType(self.t_ParEntries, dtype=TaggedFloat)
                for i in range(self.t_ParEntries):
                    l_avec[i] = TaggedFloat(l_A[i], k == 0)
                    l_bvec[i] = TaggedFloat(l_B[i], k == 0)

                l_avec1 = l_Ta.shift(l_avec)
                l_bvec1 = l_Tb.shift(l_bvec)

                l_awin.shift_right(l_avec1)
                l_bwin.shift(l_bvec1)

                if (l > 0 and k >= self.t_ParEntries + 1 and k <= self.t_ParEntries + self.t_ParEntries):
                    # for i in range(self.t_ParEntries):
                    #    l_C[i] = l_Co[i]
                    # print("l_C: ", l_C)
                    # print("l: ", l, " k: ", k)
                    # outstream.StreamToMatrix(l_Co[k - self.t_ParEntries - 1].as_list())
                    p_sum.write(l_Co[k - self.t_ParEntries - 1])
                # if l == 0:
                #    print(f"l: {l}, k: {k}, Printing Windows:")
                #    print(l_awin)
                #    print(l_bwin)
                for row in range(self.t_ParEntries):
                    l_arow = l_awin[row]
                    l_brow = l_bwin[row]
                    for col in range(self.t_ParEntries):
                        aval = l_arow[col].getFloat()
                        bval = l_brow[col].getFloat()

                        aflush = l_arow[col].getTag()
                        if aflush:
                            l_Co[row][col] = l_C[row][col]
                            l_C[row][col] = 0

                        l_C[row][col] += aval * bval
                # if l == 0:
                #     print("l_C: ", l_C)
        return p_sum
