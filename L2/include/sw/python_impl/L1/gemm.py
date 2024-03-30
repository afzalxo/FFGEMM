from xf_types import WindowRm, TriangSrl, WideType, TaggedFloat
from matrix import Matrix
from gemmMatMover import GEMMMatAMover, GEMMMatBMover, GEMMMatStreamToMatrix


class Gemm:
    def __init__(self,
                 moverA: GEMMMatAMover,
                 moverB: GEMMMatBMover,
                 p_blocks: int,
                 t_KBufferDim: int,
                 t_ParEntries: int):
        self.moverA = moverA
        self.moverB = moverB
        self.p_blocks = p_blocks
        self.t_KBufferDim = t_KBufferDim
        self.t_ParEntries = t_ParEntries

    def compute(self):
        l_awin = WindowRm(self.t_ParEntries, self.t_ParEntries, dtype=TaggedFloat)
        l_bwin = WindowRm(self.t_ParEntries, self.t_ParEntries, dtype=TaggedFloat)
        l_Ta = TriangSrl(self.t_ParEntries, dtype=TaggedFloat)
        l_Tb = TriangSrl(self.t_ParEntries, dtype=TaggedFloat)

        l_C = [WideType(self.t_ParEntries) for _ in range(self.t_ParEntries)]
        l_Co = [WideType(self.t_ParEntries) for _ in range(self.t_ParEntries)]

        outstream = GEMMMatStreamToMatrix(self.moverA.p_m, self.moverB.p_n, self.t_ParEntries)

        for l in range(self.p_blocks+1):
            for k in range(self.t_KBufferDim):
                l_A = WideType(self.t_ParEntries, dtype=int)
                l_B = WideType(self.t_ParEntries, dtype=int)

                if l < self.p_blocks:
                    l_A = next(self.moverA)
                    l_B = next(self.moverB)

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
                    # print("l_Co: ", l_Co[k - self.t_ParEntries - 1])
                    outstream.StreamToMatrix(l_Co[k - self.t_ParEntries - 1].as_list())
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
        return outstream.GetMatrix()
