from stream import Stream
from xf_types import WideType, MemIntType
from transpose import Transpose
from gemm import Gemm
from matrixBuffer import MatrixBuffer
from submatrixOps import SubMatrixOps
from buffer import GemmCBuffer


class GemmKernel:
    def __init__(
        self,
        t_FloatType: type,
        t_MemWidth: int,
        t_aColMemWords: int,
        t_aRowMemWords: int,
        t_bColMemWords: int,
    ):
        self.t_FloatType = t_FloatType
        self.t_MemWidth = t_MemWidth
        self.t_aColMemWords = t_aColMemWords
        self.t_aRowMemWords = t_aRowMemWords
        self.t_bColMemWords = t_bColMemWords

        self.t_aMH = t_MemWidth * t_aRowMemWords
        self.t_bKD = t_MemWidth * t_aColMemWords

        self.submatOpsA = SubMatrixOps(
            t_FloatType, t_MemWidth, t_aRowMemWords, t_aColMemWords
        )
        self.submatOpsB = SubMatrixOps(
            t_FloatType, t_MemWidth, t_aColMemWords, t_bColMemWords
        )
        self.submatOpsC = SubMatrixOps(
            t_FloatType, t_MemWidth, t_aRowMemWords, t_bColMemWords
        )

    def GemmReadAB(
        self,
        l_aAddr: MemIntType,
        l_bAddr: MemIntType,
        l_aColBlocks: int,
        l_aRowBlocks: int,
        l_bColBlocks: int,
        l_aWordLd: int,
        l_bWordLd: int,
        p_As: Stream,
        p_Bs: Stream,
    ):

        l_As = Stream(dtype=WideType)
        l_Bs = Stream(dtype=WideType)
        for l_aRowBlock in range(l_aRowBlocks):
            for l_bColBlock in range(l_bColBlocks):
                for l_aColBlock in range(l_aColBlocks):
                    l_As = self.submatOpsA.ReadBlock(
                        l_aAddr,
                        l_aRowBlock,
                        l_aColBlock,
                        self.t_aMH,
                        self.t_aColMemWords,
                        l_aWordLd,
                    )
                    l_Bs = self.submatOpsB.ReadBlock(
                        l_bAddr,
                        l_aColBlock,
                        l_bColBlock,
                        self.t_bKD,
                        self.t_bColMemWords,
                        l_bWordLd,
                    )
                    p_As.append(l_As)
                    p_Bs.append(l_Bs)
        return p_As, p_Bs

    def GemmWriteC(
        self,
        l_cAddr: MemIntType,
        l_cRowBlocks: int,
        l_cColBlocks: int,
        l_cWordLd: int,
        p_Cs: Stream,
    ):
        for l_cRowBlock in range(l_cRowBlocks):
            for l_cColBlock in range(l_cColBlocks):
                l_cAddr = self.submatOpsC.WriteBlock(
                    l_cAddr,
                    l_cRowBlock,
                    l_cColBlock,
                    self.t_MemWidth * self.t_aRowMemWords,
                    self.t_bColMemWords,
                    l_cWordLd,
                    p_Cs,
                )
        return l_cAddr

    def GemmBlocksStream(
        self, p_As: Stream, p_Bs: Stream, p_Cs: Stream, numBlocks: int
    ):
        l_Cs = Stream(dtype=WideType)

        p_transpBlocks = numBlocks * self.t_aRowMemWords

        p_AoutS = Stream(dtype=WideType)
        p_transposeOp = Transpose(
            self.t_FloatType,
            self.t_aColMemWords,
            self.t_MemWidth,
            self.t_MemWidth,
            p_transpBlocks,
            self.t_bColMemWords,
        )
        p_AoutS = p_transposeOp.process(p_As, p_AoutS)

        p_Bs1 = Stream(dtype=WideType)
        p_matrixBuffer = MatrixBuffer(
            self.t_FloatType, self.t_MemWidth, self.t_bKD, self.t_bColMemWords
        )
        p_Bs1 = p_matrixBuffer.process(p_Bs, p_Bs1, numBlocks, self.t_aRowMemWords)

        # print("GemmOp")
        # print(p_AoutS)
        # print(p_Bs1)
        GemmOp = Gemm(self.t_FloatType, self.t_bKD, self.t_MemWidth, self.t_MemWidth)
        l_Cs = GemmOp.gemm(
            p_AoutS, p_Bs1, numBlocks * self.t_aRowMemWords * self.t_bColMemWords
        )
        p_Cs.append(l_Cs)
        return p_Cs

    def GemmBlocks(
        self,
        p_aAddr: MemIntType,
        p_bAddr: MemIntType,
        p_cAddr: MemIntType,
        p_aColBlocks: int,
        p_aRowBlocks: int,
        p_bColBlocks: int,
        p_aLd: int,
        p_bLd: int,
        p_cLd: int,
        p_transpBlocks: int,
    ):
        l_As = Stream(dtype=WideType)
        l_Bs = Stream(dtype=WideType)
        l_Cs = Stream(dtype=WideType)
        p_cBlocks = p_aRowBlocks * p_bColBlocks
        p_abBlocks = p_cBlocks * p_aColBlocks

        l_As, l_Bs = self.GemmReadAB(
            p_aAddr,
            p_bAddr,
            p_aColBlocks,
            p_aRowBlocks,
            p_bColBlocks,
            p_aLd,
            p_bLd,
            l_As,
            l_Bs,
        )

        l_Cs = self.GemmBlocksStream(l_As, l_Bs, l_Cs, p_abBlocks)

        gemmCBuffer = GemmCBuffer(
            self.t_MemWidth,
            self.t_aRowMemWords,
            self.t_bColMemWords,
            p_aColBlocks,
            p_cBlocks,
        )
        outCs = gemmCBuffer.BufferAndDownStream(l_Cs)

        p_cAddr = self.GemmWriteC(p_cAddr, p_aRowBlocks, p_bColBlocks, p_cLd, outCs)
        return p_cAddr

    def runGemm(
        self,
        p_aAddr: MemIntType,
        p_bAddr: MemIntType,
        p_cAddr: MemIntType,
        m_M: int,
        m_N: int,
        m_K: int,
        m_Lda: int,
        m_Ldb: int,
        m_Ldc: int,
    ):
        l_aColBlocks = m_K // (self.t_MemWidth * self.t_aColMemWords)
        l_aRowBlocks = m_M // (self.t_MemWidth * self.t_aRowMemWords)
        l_bColBlocks = m_N // (self.t_MemWidth * self.t_bColMemWords)
        l_aLd = m_Lda // self.t_MemWidth
        l_bLd = m_Ldb // self.t_MemWidth
        l_cLd = m_Ldc // self.t_MemWidth

        l_transpBlocks = (
            l_aColBlocks * l_aRowBlocks * l_bColBlocks * self.t_aRowMemWords
        )
        res = self.GemmBlocks(
            p_aAddr,
            p_bAddr,
            p_cAddr,
            l_aColBlocks,
            l_aRowBlocks,
            l_bColBlocks,
            l_aLd,
            l_bLd,
            l_cLd,
            l_transpBlocks,
        )
        return res
