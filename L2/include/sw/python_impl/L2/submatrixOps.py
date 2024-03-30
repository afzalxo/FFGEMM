from xf_types import WideType, MemIntType
from stream import Stream
import matrix


class SubMatrixOps:
    def __init__(self, t_FloatType, t_MemWidth, t_RowMemWords, t_ColMemWords):
        self.t_FloatType = t_FloatType
        self.t_MemWidth = t_MemWidth
        self.t_RowMemWords = t_RowMemWords
        self.t_ColMemWords = t_ColMemWords
        self.t_MH = t_MemWidth * t_RowMemWords

    # matrixIn: input matrix in the form of List of WideTypes containing matrix elements
    def ReadBlock(
        self,
        matrixIn: MemIntType,
        l_rowInd: int,
        l_colInd: int,
        l_numElements: int,
        l_numMemWords: int,
        l_wordLd: int,
    ) -> Stream:
        p_BlockStream = Stream(dtype=WideType)
        for i in range(l_numElements):
            for j in range(l_numMemWords):
                l_SrcOffset = (
                    l_wordLd * l_numElements * l_rowInd
                    + l_colInd * l_numMemWords
                    + i * l_wordLd
                    + j
                )
                l_word = matrixIn[l_SrcOffset]
                p_BlockStream.write(l_word)

        return p_BlockStream

    def WriteBlock(
        self,
        matrixOut: MemIntType,
        l_rowInd: int,
        l_colInd: int,
        l_numElements: int,
        l_numMemWords: int,
        l_wordLd: int,
        l_outStream: Stream
    ) -> MemIntType:
        for i in range(l_numElements):
            for j in range(l_numMemWords):
                l_DstOffset = (
                    l_wordLd * l_numElements * l_rowInd
                    + l_colInd * l_numMemWords
                    + i * l_wordLd
                    + j
                )
                l_word = l_outStream.read()
                matrixOut[l_DstOffset] = l_word

        return matrixOut

    def AddBlocks(
        self,
        l_numElemsCol: int,
        l_numMemWordsRow: int,
        p_StreamA: Stream,
        p_StreamB: Stream,
    ) -> Stream:
        p_StreamOut = Stream(dtype=WideType)
        for i in range(l_numElemsCol):
            for j in range(l_numMemWordsRow):
                l_wordA = p_StreamA.read()
                l_wordB = p_StreamB.read()
                l_wordC = WideType(t_Width=self.t_MemWidth)
                for k in range(self.t_MemWidth):
                    l_wordC[k] = l_wordA[k] + l_wordB[k]
                p_StreamOut.write(l_wordC)
        return p_StreamOut

    def SubBlocks(
        self,
        l_numElemsCol: int,
        l_numMemWordsRow: int,
        p_StreamA: Stream,
        p_StreamB: Stream,
    ) -> Stream:
        p_StreamOut = Stream(dtype=WideType)
        for i in range(l_numElemsCol):
            for j in range(l_numMemWordsRow):
                l_wordA = p_StreamA.read()
                l_wordB = p_StreamB.read()
                l_wordC = WideType(t_Width=self.t_MemWidth)
                for k in range(self.t_MemWidth):
                    l_wordC[k] = l_wordA[k] - l_wordB[k]
                p_StreamOut.write(l_wordC)
        return p_StreamOut

    def AddConsecutiveBlocks(
        self,
        l_numElemsCol: int,
        l_numMemWordsRow: int,
        p_Stream: Stream
    ):
        p_StreamA = Stream(dtype=WideType)
        p_StreamA = p_Stream.split(l_numElemsCol * l_numMemWordsRow)
        p_StreamOut = self.AddBlocks(l_numElemsCol, l_numMemWordsRow, p_StreamA, p_Stream)
        return p_StreamOut, p_Stream

    def SubConsecutiveBlocks(
        self,
        l_numElemsCol: int,
        l_numMemWordsRow: int,
        p_Stream: Stream
    ):
        p_StreamA = Stream(dtype=WideType)
        p_StreamA = p_Stream.split(l_numElemsCol * l_numMemWordsRow)
        p_StreamOut = self.SubBlocks(l_numElemsCol, l_numMemWordsRow, p_StreamA, p_Stream)
        return p_StreamOut, p_Stream

    def ExtractNumWords(
        self,
        l_numMemWords: int,
        p_Stream: Stream
    ):
        p_StreamOut = Stream(dtype=WideType)
        for i in range(l_numMemWords):
            l_word = p_Stream.read()
            p_StreamOut.write(l_word)
        return p_StreamOut, p_Stream

    # For strassenSquaredKernel
    def ReadAndBufferComplete(l_Addr: MemIntType,
                              l_rowInd: int,
                              l_colInd: int,
                              l_wordLd: int,
                              l_buf: list[list[list[list[WideType]]]]) -> list[list[list[list[WideType]]]]:
        for i in range(4):
            for j in range(4):
                for k in range(self.t_MH):
                    for l in range(self.t_ColMemWords):
                        src_idx = l_rowInd * 4 * t_MH * l_wordLd + l_colInd * 4 * t_ColMemWords + i * t_MH * 4 * t_ColMemWords + j * t_ColMemWords + k * l_wordLd + l
                        read_data = l_Addr[src_idx]
                        l_buf[i][j][k][l] = read_data
        return l_buf

    def ReadAddBuffer_2(l_Addr: MemIntType,
                        l_buf0: list[list[WideType]],
                        l_buf0_sign: bool,
                        l_buf1: list[list[WideType]],
                        l_buf1_sign: bool,
                        p_BlockStream: Stream) -> Stream:
        for i in range(self.t_MH):
            for j in range(self.t_ColMemWords):
                l_word = WideType(t_Width=self.t_MemWidth)
                for k in range(self.t_MemWidth):
                    if l_buf0_sign:
                        l_word[k] += l_buf0[i][j][k]
                    else:
                        l_word[k] -= l_buf0[i][j][k]
                    if l_buf1_sign:
                        l_word[k] += l_buf1[i][j][k]
                    else:
                        l_word[k] -= l_buf1[i][j][k]
                p_BlockStream.write(l_word)


    def ReadAddBuffer_4(l_Addr: MemIntType,
                        l_buf0: list[list[list[WideType]]],
                        l_buf0_sign: bool,
                        l_buf1: list[list[list[WideType]]],
                        l_buf1_sign: bool,
                        l_buf2: list[list[list[WideType]]],
                        l_buf2_sign: bool,
                        l_buf3: list[list[list[WideType]]],
                        l_buf3_sign: bool,
                        p_BlockStream: Stream) -> Stream:
        for i in range(self.t_MH):
            for j in range(self.t_ColMemWords):
                l_word = WideType(t_Width=self.t_MemWidth)
                for k in range(self.t_MemWidth):
                    if l_buf0_sign:
                        l_word[k] += l_buf0[i][j][k]
                    else:
                        l_word[k] -= l_buf0[i][j][k]
                    if l_buf1_sign:
                        l_word[k] += l_buf1[i][j][k]
                    else:
                        l_word[k] -= l_buf1[i][j][k]
                    if l_buf2_sign:
                        l_word[k] += l_buf2[i][j][k]
                    else:
                        l_word[k] -= l_buf2[i][j][k]
                    if l_buf3_sign:
                        l_word[k] += l_buf3[i][j][k]
                    else:
                        l_word[k] -= l_buf3[i][j][k]
                p_BlockStream.write(l_word)
        return p_BlockStream

    def BlockBufferToStream(l_buf: list[list[WideType]],
                            p_BlockStream: Stream) -> Stream:
        for i in range(self.t_MH):
            for j in range(self.t_ColMemWords):
                l_word = l_buf[i][j]
                p_BlockStream.write(l_word)
        return p_BlockStream

    def WriteOutputStrassen(l_Addr: MemIntType,
                            l_rowInd: int,
                            l_colInd: int,
                            l_wordLd: int,
                            p_BlockStream: Stream) -> MemIntType:
        for i in range(4):
            for j in range(4):
                for k in range(t_MH):
                    for l in range(t_ColMemWords):
                        l_DstOffset = l_rowInd * 4 * t_MH * l_wordLd + l_colInd * 4 * t_ColMemWords + i * t_MH * 4 * t_ColMemWords + j * t_ColMemWords + k * l_wordLd + l
                        l_word = p_BlockStream.read()
                        l_Addr[l_DstOffset] = l_word
        return l_Addr
