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

#ifndef XF_BLAS_STRASSENS_KERNEL_HPP
#define XF_BLAS_STRASSENS_KERNEL_HPP

#include <cassert>
#include <iostream>
#include "types.hpp"
#include "transpose.hpp"
#include "matrixBuffer.hpp"
#include "gemm.hpp"
#include "submatrixOps.hpp"

namespace xf {

namespace blas {

/**
 * @brief Gemm class, implement C = A*B
 * t_aColMemWords defines number of memwords in the columns of one row of buffer_A. Due to the reusability, the
 height of buffer_A is only one memwords. For buffer_B, t_aColMemWords defines number of memwords in the rows of one
 column in buffer_B, t_bColMemWords defines number of memwords in the cols of one row in buffer_B. t_aRowMemWords and
 t_bColMemWords define the height and width of buffer_C in terms of memwords.
 *
 * @tparam t_FloatType matrix A, B entry data type
 * @tparam t_MemWidth number of matrix elements in one memory word
 * @tparam t_aColMemWords  number of memory words in one row of the matrix A buffer
 * @tparam t_aRowMemWords  number of memory words in one column of the matrix A buffer
 * @tparam t_bColMemWords   number of memory words in one row of the matrix B buffer
 *
 */
template <typename t_FloatType,    // matrix A, B entry data type
          unsigned int t_MemWidth, // number of matrix elements in one memory word
          unsigned int t_aColMemWords = 1, // number of memory words in one row of the matrix A buffer
          unsigned int t_aRowMemWords = 1, // number of memory words in one column of the matrix A buffer
          unsigned int t_bColMemWords = 1  // number of memory words in one row of the matrix B buffer
          >
class StrassensKernel {
   public:
    static const unsigned int t_aMH = t_MemWidth * t_aRowMemWords;  // Number of matrix elements in one column of matrix A buffer
    static const unsigned int t_bKD = t_MemWidth * t_aColMemWords;  // Number of matrix elements in one row of matrix A buffer / one col of matrix B buffer
								    
    static const unsigned int t_cMH = 2 * t_aRowMemWords * t_MemWidth;
    static const unsigned int t_strMemWords = 2 * t_bColMemWords;

    typedef WideType<t_FloatType, t_MemWidth> MemWideType;
    typedef typename MemWideType::t_TypeInt MemIntType;
    typedef hls::stream<MemIntType> MemStream;

    typedef hls::stream<typename TaggedWideType<t_FloatType, t_MemWidth>::t_TypeInt> EdgeStream;

    // type definitions for enhanced MAC implementation, using 48-bits to store accumulation results.
    typedef t_FloatType MacBitType;
    typedef MemWideType WideMacBitType;
    typedef MemStream WideMacBitStream;

    typedef GemmArgs GemmArgsType;
    SubMatrixOps<t_FloatType, t_MemWidth, t_aRowMemWords, t_aColMemWords> t_subMatOps;


   private:
    static const unsigned int t_debug = 0;

   public:
    /**
     * @brief GemmReadAB load data from matrix A, B
     *
     * @param l_aAddr  the base address of matrix A in external memory
     * @param l_bAddr  the base address of matrix B in external memory
     *
     * @param l_aColBlocks  the No. blocks along matrix X cols
     * @param l_aRowBlocks  the No. blocks along matrix X rows
     * @param l_bColBlocks  the No. blocks along matrix X cols
     *
     * @param l_aWordLd  the matrix A word leading dimention. The number of mem words in the leading dimension of matrix A. Currently low-major format.
     * @param l_bWordLd  the matrix B word leading dimention. The number of mem words in the leading dimension of matrix B
     *
     * @param p_As  the output stream for matrix A
     * @param p_Bs  the output stream for matrix B
     *
     */
    void GemmReadAB( MemIntType* l_aAddr,
                     MemIntType* l_bAddr,
                     const unsigned int l_aColBlocks,  // 64 / (16 * 4)
                     const unsigned int l_aRowBlocks,
                     const unsigned int l_bColBlocks,
                     const unsigned int l_aWordLd,   // 64 / 16
                     const unsigned int l_bWordLd,   // 64 / 16
                     MemStream& p_As,
                     MemStream& p_Bs) {
	// Load patterns follows the M1-M7 patterns in Strassens defined in Wikipedia
	// https://en.wikipedia.org/wiki/Strassen_algorithm
	const unsigned int loadPatternA[12][2] = {{0, 0}, {1, 1}, {1, 0}, {1, 1}, {0, 0}, {1, 1}, {0, 0}, {0, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}};
	const unsigned int loadPatternB[12][2] = {{0, 0}, {1, 1}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {1, 1}, {0, 0}, {0, 1}, {1, 0}, {1, 1}};
gemmreadab_outer:
        for (int l_aRowBlock = 0; l_aRowBlock < l_aRowBlocks; ++l_aRowBlock) {
gemmreadab_middle:
           for (int l_bColBlock = 0; l_bColBlock < l_bColBlocks; ++l_bColBlock) {
gemmreadab_inner:
               for (int l_aColBlock = 0; l_aColBlock < l_aColBlocks; ++l_aColBlock) {
// #pragma HLS PIPELINE OFF
gemmreadab_ldpatterns:
		    for (unsigned int i = 0; i < 12; i++) {
// #pragma HLS PIPELINE OFF
			t_subMatOps.ReadBlock(l_bAddr,
						  (l_aColBlock << 1) + loadPatternB[i][0],
						  (l_bColBlock << 1) + loadPatternB[i][1],
						  l_bWordLd,
						  p_Bs);
		        t_subMatOps.ReadBlock(l_aAddr, 
					   	  (l_aRowBlock << 1) + loadPatternA[i][0],
						  (l_aColBlock << 1) + loadPatternA[i][1],
						  l_aWordLd,
						  p_As);
			/*
		        t_subMatOps.ReadBlock(l_aAddr, 
					   	  (l_aRowBlock << 1) + loadPatternA[1][0],
						  (l_aColBlock << 1) + loadPatternA[1][1],
						  l_aWordLd,
						  p_As);
			t_subMatOps.ReadBlock(l_bAddr,
						  (l_aColBlock << 1) + loadPatternB[1][0],
						  (l_bColBlock << 1) + loadPatternB[1][1],
						  l_bWordLd,
						  p_Bs);
		        t_subMatOps.ReadBlock(l_aAddr, 
					   	  (l_aRowBlock << 1) + loadPatternA[2][0],
						  (l_aColBlock << 1) + loadPatternA[2][1],
						  l_aWordLd,
						  p_As);
		        t_subMatOps.ReadBlock(l_aAddr, 
					   	  (l_aRowBlock << 1) + loadPatternA[3][0],
						  (l_aColBlock << 1) + loadPatternA[3][1],
						  l_aWordLd,
						  p_As);
			t_subMatOps.ReadBlock(l_bAddr,
						  (l_aColBlock << 1) + loadPatternB[2][0],
						  (l_bColBlock << 1) + loadPatternB[2][1],
						  l_bWordLd,
						  p_Bs);
			*/
			/*
			t_subMatOps.ReadBlock(l_bAddr,
						  (l_aColBlock << 1) + loadPatternB[1][0],
						  (l_bColBlock << 1) + loadPatternB[1][1],
						  l_bWordLd,
						  p_Bs);
			t_subMatOps.ReadBlock(l_aAddr, 
					   	  (l_aRowBlock << 1) + loadPatternA[1][0],
						  (l_aColBlock << 1) + loadPatternA[1][1],
						  l_aWordLd,
						  p_As);
			*/
		    }
                }
            }
        }
    }
    /* See parallel implementation in L2/include/sw/python_impl/L2/strassensKernel.py */
    void GenNewLHSRHS(MemStream& p_As,
		      MemStream& p_Bs,
		      MemStream& p_LHS,
		      MemStream& p_RHS,
		      const unsigned int p_aColBlocks,
		      const unsigned int p_aRowBlocks,
		      const unsigned int p_bColBlocks) {
gen_new_outer:
    	for (int l_aRowBlock = 0; l_aRowBlock < p_aRowBlocks; ++l_aRowBlock) {
gen_new_middle:
	    for (int l_bColBlock = 0; l_bColBlock < p_bColBlocks; ++l_bColBlock) {
gen_new_inner:
	        for (int l_aColBlock = 0; l_aColBlock < p_aColBlocks; ++l_aColBlock) {
		    // t_subMatOps.AddConsecutiveBlocks(p_As, p_LHS);
		    // t_subMatOps.AddConsecutiveBlocks(p_Bs, p_RHS);
		    t_subMatOps.AddConsecutiveBlocks(p_Bs, p_RHS);
		    t_subMatOps.AddConsecutiveBlocks(p_As, p_LHS);

		    t_subMatOps.ExtractStream(p_Bs, p_RHS);
		    t_subMatOps.AddConsecutiveBlocks(p_As, p_LHS);

		    t_subMatOps.SubConsecutiveBlocks(p_Bs, p_RHS);
		    t_subMatOps.ExtractStream(p_As, p_LHS);

		    t_subMatOps.SubConsecutiveBlocks(p_Bs, p_RHS);
		    t_subMatOps.ExtractStream(p_As, p_LHS);

		    t_subMatOps.ExtractStream(p_Bs, p_RHS);
		    t_subMatOps.AddConsecutiveBlocks(p_As, p_LHS);

		    t_subMatOps.AddConsecutiveBlocks(p_Bs, p_RHS);
		    t_subMatOps.SubConsecutiveBlocks(p_As, p_LHS);

		    t_subMatOps.AddConsecutiveBlocks(p_Bs, p_RHS);
		    t_subMatOps.SubConsecutiveBlocks(p_As, p_LHS);

		    /*
		    t_subMatOps.ExtractStream(p_As, p_LHS);
		    t_subMatOps.SubConsecutiveBlocks(p_Bs, p_RHS);

		    t_subMatOps.ExtractStream(p_As, p_LHS);
		    t_subMatOps.SubConsecutiveBlocks(p_Bs, p_RHS);

		    t_subMatOps.AddConsecutiveBlocks(p_As, p_LHS);
		    t_subMatOps.ExtractStream(p_Bs, p_RHS);
		    */

		/*
		    t_subMatOps.AddConsecutiveBlocks(p_As, p_LHS);
		    t_subMatOps.AddConsecutiveBlocks(p_As, p_LHS);
		    t_subMatOps.ExtractStream(p_As, p_LHS);
		    t_subMatOps.ExtractStream(p_As, p_LHS);
		    t_subMatOps.AddConsecutiveBlocks(p_As, p_LHS);
		    // t_subMatOps.SubConsecutiveBlocks(p_As, p_LHS);
		    // t_subMatOps.SubConsecutiveBlocks(p_As, p_LHS);

		    // t_subMatOps.AddConsecutiveBlocks(p_Bs, p_RHS);
		    // t_subMatOps.AddConsecutiveBlocks(p_Bs, p_RHS);
		*/
		}
	    }
	}
    }

    void ClearBuffer(MemWideType* p_buffer) {
#pragma HLS INLINE
	for (int i = 0; i < t_cMH * t_strMemWords; i++) {
#pragma HLS PIPELINE
	    // for (int j = 0; j < t_strMemWords; j++) {
		for (int k = 0; k < t_MemWidth; k++) {
		    p_buffer[i][k] = 0;
		}
	    //}
	}
    }

    void BufferBlockStrassen(MemStream& p_stream,
		    	     MemWideType* p_buffer,
			     const unsigned int* p_ptr_blk_idx,
			     const unsigned int* p_ptr_add_or_sub,
			     const unsigned int  p_numBlocks) {
#pragma HLS INLINE
buffer_block_strassen_outermost:
        for (unsigned int i = 0; i < t_aRowMemWords; i++) {
buffer_block_strassen_middle:
	    for (unsigned int j = 0; j < t_bColMemWords; j++) {
buffer_block_strassen_inner:
		for (unsigned int l = 0; l < t_MemWidth; l++) {
// #pragma HLS PIPELINE
// #pragma HLS DEPENDENCE variable = p_buffer array inter RAW false
		    MemWideType l_val = p_stream.read();
buffer_block_strassen_inner_blk_idx:
		    for (unsigned int block_idx = 0; block_idx < p_numBlocks; block_idx++) {
#pragma HLS PIPELINE II = t_MemWidth
			    unsigned int block_row = p_ptr_blk_idx[block_idx * 2 + 0];
			    unsigned int block_col = p_ptr_blk_idx[block_idx * 2 + 1];
			    //unsigned int l_arrRowInd = l + t_MemWidth * (i + block_row * t_aRowMemWords);
			    //unsigned int l_arrColInd = j + block_col * t_bColMemWords;
			    unsigned int buffer_index = (l + t_MemWidth * (i + block_row * t_aRowMemWords)) * t_strMemWords + j + block_col * t_bColMemWords;
			    unsigned int add_or_sub = p_ptr_add_or_sub[block_idx];
			    if (add_or_sub) {
buffer_block_strassen_inner_blk_idx_add:
		    		for (unsigned int k = 0; k < t_MemWidth; k++) {
			            //p_buffer[l_arrRowInd * t_strMemWords + l_arrColInd][k] += l_val[k];
			            p_buffer[buffer_index][k] += l_val[k];
				}
			    } else {
buffer_block_strassen_inner_blk_idx_sub:
			    	for (unsigned int k = 0; k < t_MemWidth; k++) {
			            // p_buffer[l_arrRowInd * 2 * t_bColMemWords + l_arrColInd][k] -= l_val[k];
				    p_buffer[buffer_index][k] -= l_val[k];
			    	}
			    }
		    }
		}
	    }
	}
    }

    void StrassensOutBuffer(MemStream& p_istream, MemStream& p_ostream, const unsigned int p_cBlocks, const unsigned int p_aColBlocks) {
	// MemWideType l_buffer[t_cMH][t_strMemWords];
	MemWideType l_buffer[t_cMH * t_strMemWords];
out_buffer_outermost:
	for (int l_block = 0; l_block < p_cBlocks; ++l_block) {
	    ClearBuffer(&l_buffer[0]);
	    // Accumulate the blocks
out_buffer_inner1:
	    for (int m = 0; m < p_aColBlocks; ++m) {
	        // unsigned int l_ptr_blk_idx[1][2] = {{0, 0}};
		// unsigned int l_ptr_add_or_sub[1] = {1};
		// BufferBlockStrassen(p_istream, &l_buffer[0][0], &l_ptr_blk_idx[0][0], &l_ptr_add_or_sub[0], 1);
	        unsigned int l_ptr_blk_idx[2][2] = {{0, 0}, {1, 1}};
		unsigned int l_ptr_add_or_sub[2] = {1, 1};
		BufferBlockStrassen(p_istream, &l_buffer[0], &l_ptr_blk_idx[0][0], &l_ptr_add_or_sub[0], 2);

		l_ptr_blk_idx[0][0] = 1;
		l_ptr_blk_idx[0][1] = 0;
		l_ptr_blk_idx[1][0] = 1;
		l_ptr_blk_idx[1][1] = 1;

		l_ptr_add_or_sub[0] = 1;
		l_ptr_add_or_sub[1] = 0;

		BufferBlockStrassen(p_istream, &l_buffer[0], &l_ptr_blk_idx[0][0], &l_ptr_add_or_sub[0], 2);

		l_ptr_blk_idx[0][0] = 0;
		l_ptr_blk_idx[0][1] = 1;
		l_ptr_blk_idx[1][0] = 1;
		l_ptr_blk_idx[1][1] = 1;

		l_ptr_add_or_sub[0] = 1;
		l_ptr_add_or_sub[1] = 1;
		BufferBlockStrassen(p_istream, &l_buffer[0], &l_ptr_blk_idx[0][0], &l_ptr_add_or_sub[0], 2);

		l_ptr_blk_idx[0][0] = 0;
		l_ptr_blk_idx[0][1] = 0;
		l_ptr_blk_idx[1][0] = 1;
		l_ptr_blk_idx[1][1] = 0;

		l_ptr_add_or_sub[0] = 1;
		l_ptr_add_or_sub[1] = 1;

		BufferBlockStrassen(p_istream, &l_buffer[0], &l_ptr_blk_idx[0][0], &l_ptr_add_or_sub[0], 2);

		l_ptr_blk_idx[0][0] = 0;
		l_ptr_blk_idx[0][1] = 0;
		l_ptr_blk_idx[1][0] = 0;
		l_ptr_blk_idx[1][1] = 1;

		l_ptr_add_or_sub[0] = 0;
		l_ptr_add_or_sub[1] = 1;
		BufferBlockStrassen(p_istream, &l_buffer[0], &l_ptr_blk_idx[0][0], &l_ptr_add_or_sub[0], 2);
		
		unsigned int l_ptr_blk_idx1[1][2] = {{1, 1}};
		unsigned int l_ptr_add_or_sub1[1] = {1};
		BufferBlockStrassen(p_istream, &l_buffer[0], &l_ptr_blk_idx1[0][0], &l_ptr_add_or_sub1[0], 1);

		l_ptr_blk_idx1[0][0] = 0;
		l_ptr_blk_idx1[0][1] = 0;

		l_ptr_add_or_sub1[0] = 1;
		BufferBlockStrassen(p_istream, &l_buffer[0], &l_ptr_blk_idx1[0][0], &l_ptr_add_or_sub1[0], 1);
	    }
	    
	    // Stream out the buffer
out_buffer_inner2:
	    for (int i = 0; i < t_cMH * t_strMemWords; ++i) {
#pragma HLS PIPELINE
	        // for (int j = 0; j < t_strMemWords; ++j) {
		    WideMacBitType l_val = l_buffer[i];
		    p_ostream.write(l_val);
	        // }
	    }
	}
    }


    void StrassensWriteC(MemIntType* l_cAddr, MemStream& l_Cs, const unsigned int l_cRowBlocks, const unsigned int l_cColBlocks, const unsigned int l_cLd) {
	for (unsigned int l_cRowBlock = 0; l_cRowBlock < l_cRowBlocks; l_cRowBlock++) {
	    for (unsigned int l_cColBlock = 0; l_cColBlock < l_cColBlocks; l_cColBlock++) {
// #pragma HLS PIPELINE OFF
		t_subMatOps.WriteBlock(l_cAddr, 
					l_cRowBlock, 
					l_cColBlock, 
					// 2 * t_aMH, 
					// 2 * t_bColMemWords, 
					l_cLd, 
					l_Cs);
	    }
	}
    }

    void GemmSingleBlock(MemStream& p_As,
		    	 MemStream& p_Bs,
			 MemStream& p_Cs,
			 unsigned int numBlocks) {
	const unsigned int p_transpBlocks = numBlocks * t_aRowMemWords;
	const unsigned int gemm_numBlocks = numBlocks * t_aRowMemWords * t_bColMemWords;
#pragma HLS DATAFLOW
	MemStream p_Bs1, p_AoutS;
// 	EdgeStream p_AEdgeS0, p_BEdgeS0;
	// WideMacBitStream p_CEdgeS, p_COutS;
//#pragma HLS STREAM variable = p_CEdgeS depth = t_MemWidth * t_aRowMemWords * t_bColMemWords
//#pragma HLS RESOURCE variable = p_CEdgeS core = fifo_uram

	Transpose<t_FloatType, t_aColMemWords, t_MemWidth> l_transp(p_transpBlocks, t_bColMemWords);
	l_transp.process(p_As, p_AoutS);

	MatrixBuffer<typename MemWideType::t_TypeInt, t_MemWidth * t_aColMemWords, t_bColMemWords, true, false>()
	   .process(p_Bs, p_Bs1, numBlocks, t_aRowMemWords);

	Gemm<t_FloatType, t_bKD, t_MemWidth>::gemm(p_AoutS, p_Bs1, p_Cs, gemm_numBlocks);
	// Gemm<t_FloatType, t_bKD, t_MemWidth>::gemm(p_AoutS, p_Bs1, p_Cs,
	// 					  2);

	// GemmCBuffer(p_CEdgeS, 1, 1, p_Cs);
    }
    
    // load A and B in t_MemWidth x t_MemWidth size blocks, multiply blocks and write results back to memory
    void GemmBlocks(MemIntType* p_aAddr,
                    MemIntType* p_bAddr,
                    MemIntType* p_cAddr,
                    const unsigned int p_aColBlocks,
                    const unsigned int p_aRowBlocks,
                    const unsigned int p_bColBlocks,
                    const unsigned int p_aLd,
                    const unsigned int p_bLd,
                    const unsigned int p_cLd,
                    const unsigned int p_transpBlocks) {
        const unsigned int l_cBlocks = p_aRowBlocks * p_bColBlocks;
	const unsigned int l_abBlocks = l_cBlocks * p_aColBlocks;
	const unsigned int num_blocks_to_multiply = 7 * l_abBlocks;
#pragma HLS DATAFLOW

        MemStream l_As, l_Bs;
        MemStream l_Cs;
#pragma HLS STREAM variable = l_Cs depth = t_MemWidth * t_aRowMemWords * t_bColMemWords
#pragma HLS bind_storage variable = l_Cs type = fifo impl = uram

#pragma HLS STREAM variable = l_As depth = t_aColMemWords * t_MemWidth * t_aRowMemWords + 2
#pragma HLS bind_storage variable = l_As type = fifo impl = uram

#pragma HLS STREAM variable = l_Bs depth = t_aColMemWords * t_MemWidth * t_aRowMemWords + 2
#pragma HLS bind_storage variable = l_Bs type = fifo impl = uram

	MemStream l_res;
#pragma HLS STREAM variable = l_res depth = t_MemWidth * t_aRowMemWords * t_bColMemWords
#pragma HLS bind_storage variable = l_res type = fifo impl = uram

	MemStream l_lhs, l_rhs;
#pragma HLS STREAM variable = l_lhs depth = t_aColMemWords * t_MemWidth * t_aRowMemWords + 2
#pragma HLS bind_storage variable = l_lhs type = fifo impl = uram
// #pragma HLS STREAM variable = l_rhs depth = 2 * t_aColMemWords * t_MemWidth * t_aRowMemWords
// #pragma HLS bind_storage variable = l_rhs type = fifo impl = uram

	GemmReadAB(p_aAddr, p_bAddr, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_aLd, p_bLd, l_As, l_Bs);

	GenNewLHSRHS(l_As, l_Bs, l_lhs, l_rhs, p_aColBlocks, p_aRowBlocks, p_bColBlocks);

	GemmSingleBlock(l_lhs, l_rhs, l_res, num_blocks_to_multiply);
	// GemmSingleBlock(l_As, l_Bs, l_res, num_blocks_to_multiply);

	// WriteStreamToMemDebug(p_cAddr, l_res);

	StrassensOutBuffer(l_res, l_Cs, l_cBlocks, p_aColBlocks);

	StrassensWriteC(p_cAddr, l_Cs, p_aRowBlocks, p_bColBlocks, p_cLd);
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

        const unsigned int l_aColBlocks = p_Args.m_K / (t_MemWidth * t_aColMemWords * 2);  // Number of blocks along matrix A cols
        const unsigned int l_aRowBlocks = p_Args.m_M / (t_MemWidth * t_aRowMemWords * 2);
        const unsigned int l_bColBlocks = p_Args.m_N / (t_MemWidth * t_bColMemWords * 2);
        const unsigned int l_aLd = p_Args.m_Lda / t_MemWidth;  // Number of Mem words in leading dim of A
        const unsigned int l_bLd = p_Args.m_Ldb / t_MemWidth;
        const unsigned int l_cLd = p_Args.m_Ldc / t_MemWidth;
	// TODO: CHECK THIS
        unsigned int l_transpBlocks = l_aColBlocks * l_aRowBlocks * l_bColBlocks * (t_aRowMemWords * 2);
        GemmBlocks(l_aAddr, l_bAddr, l_cAddr, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd,
                   l_transpBlocks);
	// addSubMatrices(l_aAddr, l_cAddr, 0, 0, 1, 1, p_Args.m_M, p_Args.m_K);
	// t_subMatOps.AddSubmatrices(l_aAddr, l_cAddr, 0, 0, 1, 1, p_Args.m_M, p_Args.m_K);
    }

    /* /// Add two submatrices of size (t_MemWidth * t_aRowMemWords) x (t_MemWidth * t_aColMemWords)
    // taken from rowInd1, colInd1 and rowInd2, colInd2. These indices are indices of the submatrices
    // within the matrix, i.e., the matrix dims divided by 2.
    // Arguments: 
    // 		p_InAddr: base address of the matrix
    // 		p_OutAddr: base address of the output matrix
    // 		rowInd1: row index of the first submatrix
    // 		colInd1: col index of the first submatrix
    // 			...
    // 		numRows: number of rows in the full matrix
    // 		numCols: number of cols in the full matrix
    */
    void addSubMatrices(MemIntType* p_InAddr,
		    	MemIntType* p_OutAddr,
		    	unsigned int rowInd1,
			unsigned int colInd1,
			unsigned int rowInd2,
			unsigned int colInd2,
			unsigned int numRows,
			unsigned int numCols) {
	const unsigned int l_submatColBlocks = numCols / (2 * (t_MemWidth * t_aColMemWords));
	const unsigned int l_submatRowBlocks = numRows / (2 * (t_MemWidth * t_aRowMemWords));
	MemStream l_Ins1, l_Ins2, l_Outs;
	for (int l_submatRowBlock = 0; l_submatRowBlock < l_submatRowBlocks; ++l_submatRowBlock) {
	    for (int l_submatColBlock = 0; l_submatColBlock < l_submatColBlocks; ++l_submatColBlock) {
		GemmReadBlock(p_InAddr, 
			      2 * rowInd1 + l_submatRowBlock, 
			      2 * colInd1 + l_submatColBlock, 
			      t_aMH, 
			      t_aColMemWords, 
			      numCols / t_MemWidth, 
			      l_Ins1);
		GemmReadBlock(p_InAddr, 
			      2 * rowInd2 + l_submatRowBlock, 
			      2 * colInd2 + l_submatColBlock, 
			      t_aMH, 
			      t_aColMemWords, 
			      numCols / t_MemWidth, 
			      l_Ins2);
		GemmAddBlocks(t_aMH, t_aColMemWords, l_Ins1, l_Ins2, l_Outs);
	    }
	}
	GemmWriteMemStream(p_OutAddr, l_Outs, l_submatRowBlocks, l_submatColBlocks, numCols / t_MemWidth);
    }


    void runStrassensGemm(MemIntType* p_DdrRd,
		    	  MemIntType* p_DdrWr,
			  GemmArgsType& p_Args) {

	MemIntType* l_aAddr = p_DdrRd + p_Args.m_Aoffset * MemWideType::per4k();
	MemIntType* l_bAddr = p_DdrRd + p_Args.m_Boffset * MemWideType::per4k();
	MemIntType* l_cAddr = p_DdrRd + p_Args.m_Coffset * MemWideType::per4k();

        const unsigned int l_aColBlocks = p_Args.m_K / (t_MemWidth * t_aColMemWords);  // Number of blocks along matrix A cols
        const unsigned int l_aRowBlocks = p_Args.m_M / (t_MemWidth * t_aRowMemWords);
        const unsigned int l_bColBlocks = p_Args.m_N / (t_MemWidth * t_bColMemWords);
        const unsigned int l_aLd = p_Args.m_Lda / t_MemWidth;  // Number of Mem words in leading dim of A
        const unsigned int l_bLd = p_Args.m_Ldb / t_MemWidth;
        const unsigned int l_cLd = p_Args.m_Ldc / t_MemWidth;

	unsigned int l_transpBlocks = l_aColBlocks * l_aRowBlocks * l_bColBlocks * t_aRowMemWords;
	StrassensGemmBlocks(l_aAddr, l_bAddr, l_cAddr, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd, l_transpBlocks);
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

    void WriteStreamToMemDebug(MemIntType* l_cAddr,
		    	       MemStream& p_Cs
			       //unsigned int l_numMemWords) {
			       ) {
	unsigned int l_dstOffset = 0;
	for (int i = 0; i < t_aMH * t_aColMemWords; ++i) {
	    MemIntType l_val = p_Cs.read();
	    l_cAddr[l_dstOffset + i] = l_val;
	}
    }
};
}
} // namespace
#endif
