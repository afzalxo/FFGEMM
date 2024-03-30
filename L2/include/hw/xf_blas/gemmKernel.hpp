/**********
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * **********/
/**
 *  @brief GEMM header
 *
 */

#ifndef XF_BLAS_GEMM_KERNEL_HPP
#define XF_BLAS_GEMM_KERNEL_HPP

#include <cassert>
#include <iostream>
#include "types.hpp"
#include "transpose.hpp"
#include "matrixBuffer.hpp"
#include "gemm.hpp"

namespace xf {

namespace blas {

/**
 * @brief Gemm class, implement C = A*B+X
 * t_aColMemWords defines number of memwords in the columns of one row of buffer_A. Due to the reusability, the
 height of buffer_A is only one memwords. For buffer_B, t_aColMemWords defines number of memwords in the rows of one
 column in buffer_B, t_bColMemWords defines number of memwords in the cols of one row in buffer_B. t_aRowMemWords and
 t_bColMemWords define the height and width of buffer_C in terms of memwords.
 *
 * @tparam t_FloatType matrix A, B entry data type
 * @tparam t_XDataType matrix X entry data type
 * @tparam t_MemWidth number of matrix elements in one memory word
 * @tparam t_XMemWidth number of matrix X elements in one memory word
 * @tparam t_aColMemWords  number of memory words in one row of the matrix A buffer
 * @tparam t_aRowMemWords  number of memory words in one column of the matrix A buffer
 * @tparam t_bColMemWords   number of memory words in one row of the matrix B buffer
 *
 */
template <typename t_FloatType,    // matrix A, B entry data type
          typename t_XDataType,    // matrix X entry data type
          unsigned int t_MemWidth, // number of matrix elements in one memory word
          unsigned int t_XMemWidth,
          unsigned int t_aColMemWords = 1, // number of memory words in one row of the matrix A buffer
          unsigned int t_aRowMemWords = 1, // number of memory words in one column of the matrix A buffer
          unsigned int t_bColMemWords = 1  // number of memory words in one row of the matrix B buffer
          >
class GemmKernel {
   public:
    static const unsigned int t_aMH = t_MemWidth * t_aRowMemWords;  // Number of matrix elements in one column of matrix A buffer
    static const unsigned int t_bKD = t_MemWidth * t_aColMemWords;  // Number of matrix elements in one row of matrix A buffer / one col of matrix B buffer
    static const unsigned int t_MemOverXMem = t_MemWidth / t_XMemWidth;
    static const unsigned int t_xColMemWords = t_bColMemWords * t_MemOverXMem;

    typedef WideType<t_FloatType, t_MemWidth> MemWideType;
    typedef typename MemWideType::t_TypeInt MemIntType;
    typedef hls::stream<MemIntType> MemStream;
    typedef TaggedWideType<t_FloatType, t_MemWidth> TaggedWideFloat;

    typedef hls::stream<typename TaggedWideType<t_FloatType, t_MemWidth>::t_TypeInt> EdgeStream;

    typedef WideType<t_XDataType, t_XMemWidth> XMemWideType;
    typedef hls::stream<typename WideType<t_XDataType, t_XMemWidth>::t_TypeInt> XMemStream;
    typedef WideType<t_XDataType, t_MemWidth> MemWideTypeForX;

    // type definitions for enhanced MAC implementation, using 48-bits to store accumulation results.
    typedef t_FloatType MacBitType;
    typedef MemWideType WideMacBitType;
    typedef MemStream WideMacBitStream;

    typedef GemmArgs GemmArgsType;

   private:
    static const unsigned int t_debug = 0;

   public:
    /**
     * @brief GemmReadMatA load data from Matrix A
     *
     * @param l_aAddr  the base address of matrix A in external memory
     * @param l_aColBlocks the No. blocks along matrix A cols
     * @param l_aRowBlocks the No. blocks along matrix A rows
     * @param l_bColBlocks the No. blocks along matrix B cols
     * @param l_aWordLd the matrix A word leading dimention
     * @param p_As the output stream
     *
     */
//     void GemmReadMatA(MemIntType* l_aAddr,
//                       unsigned int l_aColBlocks,
//                       unsigned int l_aRowBlocks,
//                       unsigned int l_bColBlocks,
//                       unsigned int l_aWordLd,
//                       MemStream& p_As) {
//         assert(t_MemOverXMem != 0);
//         assert(t_MemOverXMem * t_XMemWidth == t_MemWidth);
//         for (int l_aRowBlock = 0; l_aRowBlock < l_aRowBlocks; ++l_aRowBlock) {
//             for (int l_bColBlock = 0; l_bColBlock < l_bColBlocks; ++l_bColBlock) {
//                 for (int l_aColBlock = 0; l_aColBlock < l_aColBlocks; ++l_aColBlock) {
//                     for (int i = 0; i < t_aMH; ++i) {
//                         //#pragma HLS PIPELINE II = t_aColMemWords
//                         for (int j = 0; j < t_aColMemWords; ++j) {
// #pragma HLS PIPELINE
//                             unsigned int l_aSrcOffset =
//                                 l_aWordLd * t_aMH * l_aRowBlock + l_aColBlock * t_aColMemWords + i * l_aWordLd + j;
//                             MemIntType l_word = l_aAddr[l_aSrcOffset];
//                             p_As.write(l_word);
//                         }
//                     }
//                 }
//             }
//         }
//     }

    /**
     * @brief GemmReadMatB load data from matrix B
     *
     * @param l_bAddr  the base address of matrix B in external memory
     * @param l_aColBlocks  the No. blocks along matrix B cols
     * @param l_aRowBlocks  the No. blocks along matrix B rows
     * @param l_bColBlocks  the No. blocks along matrix B cols
     * @param l_bWordLd  the matrix B word leading dimention
     * @param p_Bs  the output stream
     *
     */
//     void GemmReadMatB(MemIntType* l_bAddr,
//                       unsigned int l_aColBlocks,
//                       unsigned int l_aRowBlocks,
//                       unsigned int l_bColBlocks,
//                       unsigned int l_bWordLd,
//                       MemStream& p_Bs) {
//         assert(t_MemOverXMem != 0);
//         assert(t_MemOverXMem * t_XMemWidth == t_MemWidth);

//         for (int l_aRowBlock = 0; l_aRowBlock < l_aRowBlocks; ++l_aRowBlock) {
//             for (int l_bColBlock = 0; l_bColBlock < l_bColBlocks; ++l_bColBlock) {
//                 for (int l_aColBlock = 0; l_aColBlock < l_aColBlocks; ++l_aColBlock) {
//                     for (int i = 0; i < t_bKD; ++i) {
// #pragma HLS PIPELINE II = t_bColMemWords
//                         for (int j = 0; j < t_bColMemWords; ++j) {
//                             unsigned int l_bSrcOffset =
//                                 i * l_bWordLd + l_bWordLd * t_bKD * l_aColBlock + l_bColBlock * t_bColMemWords + j;
//                             MemIntType l_word = l_bAddr[l_bSrcOffset];
//                             p_Bs.write(l_word);
//                         }
//                     }
//                 }
//             }
//         }
//     }
    /**
     * @brief GemmReadMatX load data from matrix X
     *
     * @param l_xAddr  the base address of matrix X in external memory
     * @param l_aColBlocks  the No. blocks along matrix X cols
     * @param l_aRowBlocks  the No. blocks along matrix X rows
     * @param l_bColBlocks  the No. blocks along matrix X cols
     * @param l_xWordLd  the matrix X word leading dimention
     * @param p_Xs  the output stream
     *
     */
//     void GemmReadMatX(MemIntType* l_xAddr,
//                       unsigned int l_aColBlocks,
//                       unsigned int l_aRowBlocks,
//                       unsigned int l_bColBlocks,
//                       unsigned int l_xWordLd,
//                       XMemStream& p_Xs) {
//         assert(t_MemOverXMem != 0);
//         assert(t_MemOverXMem * t_XMemWidth == t_MemWidth);

//         for (int l_aRowBlock = 0; l_aRowBlock < l_aRowBlocks; ++l_aRowBlock) {
//             for (int l_bColBlock = 0; l_bColBlock < l_bColBlocks; ++l_bColBlock) {
//                 WideConv<MemWideType, XMemWideType> l_conv;
// #if BLAS_XVEC
//                 for (int j = 0; j < t_xColMemWords; ++j) {
// #pragma HLS PIPELINE
//                     MemIntType l_word = l_xAddr[l_bColBlock * t_bColMemWords * t_MemOverXMem + j];
//                     XMemWideType l_wordx = l_conv.convert(l_word);
//                     p_Xs.write(l_wordx);
//                 }
// #else
//                 for (int i = 0; i < t_aMH; ++i) {
// #pragma HLS PIPELINE II = t_xColMemWords
//                     for (int j = 0; j < t_xColMemWords; ++j) {
//                         unsigned int l_xSrcOffset = l_xWordLd * t_aMH * l_aRowBlock +
//                                                     l_bColBlock * t_bColMemWords * t_MemOverXMem + l_xWordLd * i + j;
//                         MemIntType l_word = l_xAddr[l_xSrcOffset];
//                         XMemWideType l_wordx = l_conv.convert(l_word);
//                         p_Xs.write(l_wordx);
//                     }
//                 }
// #endif
//             }
//         }
//     }
    /**
     * @brief GemmReadABX load data from matrix A, B and X
     *
     * @param l_aAddr  the base address of matrix A in external memory
     * @param l_bAddr  the base address of matrix B in external memory
     * @param l_xAddr  the base address of matrix X in external memory
     *
     * @param l_aColBlocks  the No. blocks along matrix X cols
     * @param l_aRowBlocks  the No. blocks along matrix X rows
     * @param l_bColBlocks  the No. blocks along matrix X cols
     *
     * @param l_aWordLd  the matrix A word leading dimention. The number of mem words in the leading dimension of matrix A. Currently low-major format.
     * @param l_bWordLd  the matrix B word leading dimention. The number of mem words in the leading dimension of matrix B
     * @param l_xWordLd  the matrix X word leading dimention. The number of mem words in the leading dimension of matrix X
     *
     * @param p_As  the output stream for matrix A
     * @param p_Bs  the output stream for matrix B
     * @param p_Xs  the output stream for matrix X
     *
     */
    void GemmReadABX(MemIntType* l_aAddr,
                     MemIntType* l_bAddr,
                     MemIntType* l_xAddr,
                     unsigned int l_aColBlocks,  // 64 / (16 * 4)
                     unsigned int l_aRowBlocks,
                     unsigned int l_bColBlocks,
                     unsigned int l_aWordLd,   // 64 / 16
                     unsigned int l_bWordLd,   // 64 / 16
                     unsigned int l_xWordLd,
                     MemStream& p_As,
                     MemStream& p_Bs,
                     XMemStream& p_Xs) {
#ifndef __SYNTHESIS__
        assert(t_MemOverXMem != 0);
        assert(t_MemOverXMem * t_XMemWidth == t_MemWidth);
#endif
        for (int l_aRowBlock = 0; l_aRowBlock < l_aRowBlocks; ++l_aRowBlock) {
            for (int l_bColBlock = 0; l_bColBlock < l_bColBlocks; ++l_bColBlock) {
                for (int l_aColBlock = 0; l_aColBlock < l_aColBlocks; ++l_aColBlock) {
                    // l_bufferB
                    for (int i = 0; i < t_bKD; ++i)  // Number of matrix elements in one col of matrix B buffer
#pragma HLS PIPELINE II = t_bColMemWords
                        for (int j = 0; j < t_bColMemWords; j++) {  // number of memory words in one row of the matrix B buffer
                            unsigned int l_bSrcOffset =
                                i * l_bWordLd + l_bWordLd * t_bKD * l_aColBlock + l_bColBlock * t_bColMemWords + j;
                            MemIntType l_word = l_bAddr[l_bSrcOffset];
                            p_Bs.write(l_word);
                        }
                    // l_bufferA
                    for (int i = 0; i < t_aMH; i++)  // Number of matrix elements in one col of matrix A buffer
#pragma HLS PIPELINE II = t_aColMemWords
                        for (int j = 0; j < t_aColMemWords; j++) {
                            unsigned int l_aSrcOffset =
                                l_aWordLd * t_aMH * l_aRowBlock + l_aColBlock * t_aColMemWords + i * l_aWordLd + j;
                            MemIntType l_word = l_aAddr[l_aSrcOffset];
                            p_As.write(l_word);
                        }
                }
                // read X block
                WideConv<MemWideType, XMemWideType> l_conv;
#if BLAS_XVEC
                for (int j = 0; j < t_xColMemWords; ++j) {
#pragma HLS PIPELINE
                    unsigned int l_xSrcOffset =
                        l_xWordLd * t_aMH * l_aRowBlock + l_bColBlock * t_bColMemWords * t_MemOverXMem + j;
                    MemIntType l_word = l_xAddr[l_xSrcOffset];
                    XMemWideType l_wordx = l_conv.convert(l_word);
                    p_Xs.write(l_wordx);
                }
#else
                for (int i = 0; i < t_aMH; ++i)
#pragma HLS PIPELINE II = t_xColMemWords
                    for (int j = 0; j < t_xColMemWords; j++) {
                        unsigned int l_xSrcOffset = l_xWordLd * t_aMH * l_aRowBlock +
                                                    l_bColBlock * t_bColMemWords * t_MemOverXMem + l_xWordLd * i + j;
                        MemIntType l_word = l_xAddr[l_xSrcOffset];
                        XMemWideType l_wordx = l_conv.convert(l_word);
                        p_Xs.write(l_wordx);
                    }
#endif
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // GEMM C Buffering
    //
    ///////////////////////////////////////////////////////////////////////////
    void GemmCBuffer(WideMacBitStream& p_Cs,
                     unsigned int p_aColBlocks,
                     unsigned int p_cBlocks,
                     WideMacBitStream& p_Cout) {
        WideMacBitType l_bufferC[t_aMH * t_bColMemWords];

        for (int i = 0; i < t_aMH * t_bColMemWords; i++)
#pragma HLS PIPELINE
            for (int j = 0; j < t_MemWidth; j++) l_bufferC[i][j] = 0;

        for (int l_block = 0; l_block < p_cBlocks; ++l_block) {
            for (int m = 0; m < p_aColBlocks; ++m) {
                for (int i = 0; i < t_aRowMemWords; ++i) {
                    for (int j = 0; j < t_bColMemWords; ++j) {
                        for (int l = 0; l < t_MemWidth; ++l) {
#pragma HLS DEPENDENCE variable = l_bufferC array inter RAW false
#pragma HLS PIPELINE
                            unsigned int l_arrIdx = (l + i * t_MemWidth) * t_bColMemWords + j;
                            WideMacBitType l_val = p_Cs.read();
                            for (int k = 0; k < t_MemWidth; ++k) {
                                l_bufferC[l_arrIdx][k] += l_val[k];
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < t_bColMemWords * t_aRowMemWords * t_MemWidth; ++i) {
#pragma HLS PIPELINE
                WideMacBitType l_val = l_bufferC[i];
                p_Cout.write(l_val);
                for (int k = 0; k < t_MemWidth; k++) l_bufferC[i][k] = 0;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // GEMM Add X
    //
    ///////////////////////////////////////////////////////////////////////////
    void GemmAddX(
        WideMacBitStream& p_Cs, XMemStream& p_Xs, unsigned int p_cBlocks, int32_t p_postScale, MemStream& p_Cout) {
#if BLAS_XVEC
        MemWideTypeForX l_bufferX[t_bColMemWords];
#else
        MemWideTypeForX l_bufferX[t_aMH * t_bColMemWords];
#endif

        ap_uint<32> l_postScale = p_postScale;
        ap_uint<16> l_postScaleVal = l_postScale.range(23, 8);
        ap_uint<8> l_postScaleShift = l_postScale.range(7, 0);

        for (int l_block = 0; l_block < p_cBlocks; ++l_block) {
// read
#if BLAS_XVEC
            for (int xc = 0; xc < t_bColMemWords; ++xc) {
                MemWideTypeForX l_wideWordX;
                for (int xw = 0; xw < t_MemOverXMem; ++xw) {
                    // #pragma HLS PIPELINE
                    XMemWideType l_wordX = p_Xs.read();
                    for (int xi = 0; xi < t_XMemWidth; ++xi) {
                        l_wideWordX[xw * t_XMemWidth + xi] = l_wordX[xi];
                    }
                }
                l_bufferX[xc] = l_wideWordX;
            }
#else
            for (int xr = 0; xr < t_aMH; ++xr) {
                for (int xc = 0; xc < t_bColMemWords; ++xc) {
                    MemWideTypeForX l_wideWordX;
                    for (int xw = 0; xw < t_MemOverXMem; ++xw) {
#pragma HLS PIPELINE
                        XMemWideType l_wordX = p_Xs.read();
                        for (int xi = 0; xi < t_XMemWidth; ++xi) {
                            l_wideWordX[xw * t_XMemWidth + xi] = l_wordX[xi];
                        }
                    }
                    l_bufferX[xr * t_bColMemWords + xc] = l_wideWordX;
                }
            }
#endif
            for (int i = 0; i < t_aRowMemWords * t_MemWidth; ++i) {
                for (int j = 0; j < t_bColMemWords; ++j) {
                    WideMacBitType l_val = p_Cs.read();
#if BLAS_XVEC
                    MemWideTypeForX l_xVal = l_bufferX[j];
#else
                    MemWideTypeForX l_xVal = l_bufferX[i * t_bColMemWords + j];
#endif
                    MemWideType l_cWord;
#pragma HLS PIPELINE
                    for (int w = 0; w < t_MemWidth; ++w) {
                        t_FloatType l_cEntry;
                        l_cEntry = l_val[w] + l_xVal[w];

                        l_cWord[w] = l_cEntry;
                    }
                    p_Cout.write(l_cWord);
                }
            }
        }
    }

    void GemmBlockStream(MemStream& p_As,
                         MemStream& p_Bs,
                         XMemStream& p_Xs,
                         MemStream& p_Cs,
                         unsigned int p_aColBlocks,
                         unsigned int p_aRowBlocks,
                         unsigned int p_bColBlocks,
                         unsigned int p_transpBlocks,
                         int32_t p_postScale) {
        unsigned int l_cBlocks = p_aRowBlocks * p_bColBlocks;
        unsigned int l_abBlocks = l_cBlocks * p_aColBlocks;
#pragma HLS DATAFLOW

        MemStream p_Bs1, p_AoutS, p_CBufferS;
        EdgeStream p_AEdgeS0, p_BEdgeS0;
        WideMacBitStream p_CEdgeS, p_COutS;
#pragma HLS STREAM variable = p_CEdgeS depth = t_MemWidth * t_aRowMemWords * t_bColMemWords
#pragma HLS RESOURCE variable = p_CEdgeS core = fifo_uram

        Transpose<t_FloatType, t_aColMemWords, t_MemWidth> l_transp(p_transpBlocks, t_bColMemWords);  // (32, 4)
        l_transp.process(p_As, p_AoutS);

        MatrixBuffer<typename MemWideType::t_TypeInt, t_MemWidth * t_aColMemWords, t_bColMemWords, true, false>()
            .process(p_Bs, p_Bs1, l_abBlocks, t_aRowMemWords);

        Gemm<t_FloatType, t_bKD, t_MemWidth>::gemm(p_AoutS, p_Bs1, p_CEdgeS,
                                                   l_abBlocks * t_aRowMemWords * t_bColMemWords);

        GemmCBuffer(p_CEdgeS, p_aColBlocks, l_cBlocks, p_COutS);
        GemmAddX(p_COutS, p_Xs, l_cBlocks, p_postScale, p_Cs);
    }
//     void GemmReadAndMult(MemIntType* p_aAddr,
//                          MemIntType* p_bAddr,
//                          MemIntType* p_xAddr,
//                          unsigned int p_aColBlocks,
//                          unsigned int p_aRowBlocks,
//                          unsigned int p_bColBlocks,
//                          unsigned int p_aLd,
//                          unsigned int p_bLd,
//                          unsigned int p_xLd,
//                          unsigned int p_transpBlocks,
//                          int32_t p_postScale,
//                          MemStream& p_Cs) {
// #pragma HLS DATAFLOW

//         MemStream l_As, l_Bs;
//         XMemStream l_Xs;

// #if BLAS_CACHE
//         // GemmReadMatA(p_aAddr, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_aLd, l_As);
//         // GemmReadMatB(p_bAddr, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_bLd, l_Bs);
//         // GemmReadMatX(p_xAddr, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_xLd, l_Xs);
// #else
// #pragma HLS STREAM variable = l_As depth = t_aColMemWords * t_MemWidth * t_aRowMemWords
// #pragma HLS RESOURCE variable = l_As core = fifo_uram
//         GemmReadABX(p_aAddr, p_bAddr, p_xAddr, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_aLd, p_bLd, p_xLd, l_As,
//                     l_Bs, l_Xs);
// #endif
//         GemmBlockStream(l_As, l_Bs, l_Xs, p_Cs, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_transpBlocks, p_postScale);
//     }
    // load A and B in t_MemWidth x t_MemWidth size blocks, multiply blocks and write results back to memory
    void GemmBlocks(MemIntType* p_aAddr,
                    MemIntType* p_bAddr,
                    MemIntType* p_cAddr,
                    MemIntType* p_xAddr,
                    unsigned int p_aColBlocks,
                    unsigned int p_aRowBlocks,
                    unsigned int p_bColBlocks,
                    unsigned int p_aLd,
                    unsigned int p_bLd,
                    unsigned int p_cLd,
                    unsigned int p_xLd,
                    unsigned int p_transpBlocks,
                    int32_t p_postScale) {
#pragma HLS DATAFLOW

        MemStream l_As, l_Bs;
        XMemStream l_Xs;
        MemStream l_Cs;
#pragma HLS STREAM variable = l_Cs depth = t_MemWidth * t_aRowMemWords * t_bColMemWords
#pragma HLS RESOURCE variable = l_Cs core = fifo_uram

#pragma HLS STREAM variable = l_As depth = t_aColMemWords * t_MemWidth * t_aRowMemWords
#pragma HLS RESOURCE variable = l_As core = fifo_uram

        unsigned int l_cBlocks = p_aRowBlocks * p_bColBlocks;

        GemmReadABX(p_aAddr, p_bAddr, p_xAddr, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_aLd, p_bLd, p_xLd, l_As,
                    l_Bs, l_Xs);
        GemmBlockStream(l_As, l_Bs, l_Xs, l_Cs, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_transpBlocks, p_postScale);
        GemmWriteMemStream(p_cAddr, l_Cs, p_aRowBlocks, p_bColBlocks, p_cLd);
    }

    /**
     * @brief runGemm launch gemm operation with given arguments
     *
     * @param p_DdrRd  the DDR/HBM address for input data
     * @param p_DdrWr  the DDR/HBM address for output data
     * @param p_Args  the arguments for gemm
     *
     */
    void runGemm(MemIntType* p_DdrRd, // base DDR/memory address for matrix A and B
                 MemIntType* p_DdrWr, // base DDR/memory address for matrix C
                 GemmArgsType& p_Args // GEMM argument that stores the address offset of matrix A, B and C, sizes of
                 // matrix dimensions (M, K and N) and lead dimension sizes for matrix A, B and C
                 ) {
        MemIntType* l_aAddr = p_DdrRd + p_Args.m_Aoffset * MemWideType::per4k();
        MemIntType* l_bAddr = p_DdrRd + p_Args.m_Boffset * MemWideType::per4k();
        MemIntType* l_xAddr = p_DdrRd + p_Args.m_Xoffset * MemWideType::per4k();
        MemIntType* l_cAddr = p_DdrWr + p_Args.m_Coffset * MemWideType::per4k();

        const unsigned int l_aColBlocks = p_Args.m_K / (t_MemWidth * t_aColMemWords);  // Number of blocks along matrix A cols
        const unsigned int l_aRowBlocks = p_Args.m_M / (t_MemWidth * t_aRowMemWords);
        const unsigned int l_bColBlocks = p_Args.m_N / (t_MemWidth * t_bColMemWords);
        const unsigned int l_aLd = p_Args.m_Lda / t_MemWidth;  // Number of Mem words in leading dim of A
        const unsigned int l_bLd = p_Args.m_Ldb / t_MemWidth;
        const unsigned int l_cLd = p_Args.m_Ldc / t_MemWidth;
        const unsigned int l_xLd = p_Args.m_Ldx / t_XMemWidth;
        const int32_t l_postScale = p_Args.m_postScale;
        unsigned int l_transpBlocks = l_aColBlocks * l_aRowBlocks * l_bColBlocks * t_aRowMemWords;
        GemmBlocks(l_aAddr, l_bAddr, l_cAddr, l_xAddr, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd,
                   l_xLd, l_transpBlocks, l_postScale);
    }

    /**
     * @brief GemmWriteMemStream write matrix data to Memory
     *
     * @param l_cAddr  the base address of matrix in external memory
     * @param p_Cs  the input stream
     * @param l_aRowBlocks  the No. blocks along matrix X rows
     * @param l_bColBlocks  the No. blocks along matrix X cols
     * @param l_cWordLd  the matrix word leading dimention
     *
     */
    void GemmWriteMemStream(MemIntType* l_cAddr,
                            MemStream& p_Cs,
                            unsigned int l_aRowBlocks,
                            unsigned int l_bColBlocks,
                            unsigned int l_cWordLd) {
        unsigned int l_rowOffset = 0;
        unsigned int l_colOffset = 0;

        for (int rowBlock = 0; rowBlock < l_aRowBlocks; ++rowBlock) {
            for (int colBlock = 0; colBlock < l_bColBlocks; ++colBlock) {
                for (int i = 0; i < t_aRowMemWords * t_MemWidth; ++i)
#pragma HLS PIPELINE II = t_bColMemWords
                    for (int j = 0; j < t_bColMemWords; j++) {
                        unsigned int l_dstOffset = i * l_cWordLd + l_cWordLd * t_MemWidth * t_aRowMemWords * rowBlock +
                                                   colBlock * t_bColMemWords;
                        MemIntType l_val = p_Cs.read();
                        l_cAddr[l_dstOffset + j] = l_val;
                    }
            }
        }
    }
};
}
} // namespace
#endif
