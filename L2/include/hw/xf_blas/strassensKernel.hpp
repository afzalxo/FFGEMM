#ifndef XF_BLAS_STRASSENSKERNEL_HPP
#define XF_BLAS_STRASSENSKERNEL_HPP

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
 * @brief StrassensKernel class, implement C = A*B using the Strassens Algorithm for 2x2 submatrices. This kernel uses
 * a single level of recursion.
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
								    
    typedef WideType<t_FloatType, t_MemWidth> MemWideType;
    typedef typename MemWideType::t_TypeInt MemIntType;
    typedef hls::stream<MemIntType> MemStream;

    typedef hls::stream<typename TaggedWideType<t_FloatType, t_MemWidth>::t_TypeInt> EdgeStream;

    // type definitions for enhanced MAC implementation, using 48-bits to store accumulation results.
    typedef t_FloatType MacBitType;
    typedef MemWideType WideMacBitType;
    typedef MemStream WideMacBitStream;

    typedef GemmArgs GemmArgsType;
    static const unsigned int t_StrassensFactor = 2;
    SubMatrixOps<t_FloatType, t_MemWidth, t_aRowMemWords, t_aColMemWords, t_StrassensFactor> t_subMatOps;


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
     * @param s_lhs  the output stream for LHS submatrix
     * @param s_rhs  the output stream for RHS submatrix
     *
     */
    void GemmReadAB( MemIntType* l_aAddr,
                     MemIntType* l_bAddr,
                     const unsigned int l_aColBlocks,
                     const unsigned int l_aRowBlocks,
                     const unsigned int l_bColBlocks,
                     const unsigned int l_aWordLd,
                     const unsigned int l_bWordLd,
                     MemStream& s_lhs,
		     MemStream& s_rhs) {
	MemWideType buffer_a[t_StrassensFactor*t_StrassensFactor*t_aMH*t_aColMemWords];
// #pragma HLS BIND_STORAGE variable = buffer_a type = RAM_2P impl = BRAM
	MemWideType buffer_b[t_StrassensFactor*t_StrassensFactor*t_aMH*t_aColMemWords];
// #pragma HLS BIND_STORAGE variable = buffer_b type = RAM_2P impl = BRAM

gemmreadab_outer:
        for (int l_aRowBlock = 0; l_aRowBlock < l_aRowBlocks; ++l_aRowBlock) {
gemmreadab_middle:
           for (int l_bColBlock = 0; l_bColBlock < l_bColBlocks; ++l_bColBlock) {
gemmreadab_inner:
              for (int l_aColBlock = 0; l_aColBlock < l_aColBlocks; ++l_aColBlock) {
gemmreadab_ldpatterns:
		t_subMatOps.ReadToBuffer(l_bAddr, l_aColBlock, l_bColBlock, l_bWordLd, buffer_b);
		t_subMatOps.ReadToBuffer(l_aAddr, l_aRowBlock, l_aColBlock, l_aWordLd, buffer_a);
		t_subMatOps.AddBlocks_2(buffer_b,
				0, 0, 1,
				1, 1, 1,
				s_rhs);
		t_subMatOps.AddBlocks_2(buffer_a,
				0, 0, 1,
				1, 1, 1,
				s_lhs);
		t_subMatOps.BufferToStream(buffer_b, 0, 0, s_rhs);
		t_subMatOps.AddBlocks_2(buffer_a,
				1, 0, 1,
				1, 1, 1,
				s_lhs);
		t_subMatOps.AddBlocks_2(buffer_b,
				0, 1, 1,
				1, 1, 0,
				s_rhs);
		t_subMatOps.BufferToStream(buffer_a, 0, 0, s_lhs);
		t_subMatOps.AddBlocks_2(buffer_b,
				1, 0, 1,
				0, 0, 0,
				s_rhs);	
		t_subMatOps.BufferToStream(buffer_a, 1, 1, s_lhs);
		t_subMatOps.BufferToStream(buffer_b, 1, 1, s_rhs);
		t_subMatOps.AddBlocks_2(buffer_a,
				0, 0, 1,
				0, 1, 1,
				s_lhs);
		t_subMatOps.AddBlocks_2(buffer_b,
				0, 0, 1,
				0, 1, 1,
				s_rhs);
		t_subMatOps.AddBlocks_2(buffer_a,
				1, 0, 1,
				0, 0, 0,
				s_lhs);
		t_subMatOps.AddBlocks_2(buffer_b,
				1, 0, 1,
				1, 1, 1,
				s_rhs);
		t_subMatOps.AddBlocks_2(buffer_a,
				0, 1, 1,
				1, 1, 0,
				s_lhs);
	      }
           }
        }
    }

    void ClearBuffer(MemWideType p_buffer[t_StrassensFactor][t_StrassensFactor][t_aColMemWords*t_aMH]) {
#pragma HLS INLINE
clear_buffer_outermost:
	for (unsigned int i = 0; i < t_StrassensFactor; i++) {
	    for (unsigned int j = 0; j < t_StrassensFactor; j++) {
		    for (unsigned int k = 0; k < t_aMH; k++) {
#pragma HLS PIPELINE II = 1
		for (unsigned int l = 0; l < t_aColMemWords; l++) {
			unsigned int buffer_idx = l * t_aMH + k;
		    	for (unsigned int m = 0; m < t_MemWidth; m++) {
			    p_buffer[i][j][buffer_idx][m] = 0;
			}
		}
	            }
	    }
	}
    }

    void BufferBlockStrassen_2(MemStream& p_stream,
		    	       MemWideType p_buffer0[t_aColMemWords*t_aMH],
			       const bool p_buffer0_sign,
		    	       MemWideType p_buffer1[t_aColMemWords*t_aMH],
			       const bool p_buffer1_sign) {
buffer_block_strassen_4_outermost:
	for (unsigned int i = 0; i < t_aRowMemWords; i++) {
		    const unsigned int buffer_dst_idx0 = i * t_aMH;
buffer_block_strassen_4_inner:
	    for (unsigned int j = 0; j < t_aMH; j++) {
#pragma HLS PIPELINE II = t_aRowMemWords
buffer_block_strassen_4_inner_blk_idx:
		    MemWideType l_val = p_stream.read();
		    for (unsigned int k = 0; k < t_MemWidth; k++) {
			if (p_buffer0_sign) {
			   p_buffer0[buffer_dst_idx0+j][k] += l_val[k];
			} else {
			   p_buffer0[buffer_dst_idx0+j][k] -= l_val[k];
			}
			if (p_buffer1_sign) {
			   p_buffer1[buffer_dst_idx0+j][k] += l_val[k];
			} else {
			   p_buffer1[buffer_dst_idx0+j][k] -= l_val[k];
			}
		    }
	    }
	}
    }

    void BufferBlockStrassen_1(MemStream& p_stream,
		    	       		 MemWideType p_buffer[t_aColMemWords*t_aMH],
			       		 const bool p_buffer0_sign) {
buffer_block_strassen_4_outermost:
	for (unsigned int i = 0; i < t_aRowMemWords; i++) {
	    const unsigned int buffer_dst_idx0 = i * t_aMH;
buffer_block_strassen_4_inner:
	    for (unsigned int j = 0; j < t_aMH; j++) {
#pragma HLS PIPELINE II = t_aRowMemWords
buffer_block_strassen_4_inner_blk_idx:
		    MemWideType l_val = p_stream.read();
		    for (unsigned int k = 0; k < t_MemWidth; k++) {
			if (p_buffer0_sign) {
			   p_buffer[buffer_dst_idx0+j][k] += l_val[k];
			} else {
			   p_buffer[buffer_dst_idx0+j][k] -= l_val[k];
			}
		    }
	    }
	}
    }

    void StrassensOutBuffer(MemStream& p_istream, MemStream& p_ostream, const unsigned int p_cBlocks, const unsigned int p_aColBlocks) {
	// MemWideType l_buffer[t_StrassensFactor][t_StrassensFactor][t_aMH][t_aColMemWords];
	// MemWideType l_buffer[t_StrassensFactor][t_StrassensFactor][t_aColMemWords][t_aMH];
	MemWideType l_buffer[t_StrassensFactor][t_StrassensFactor][t_aColMemWords*t_aMH];
// #pragma HLS BIND_STORAGE variable = l_buffer type = RAM_2P impl = BRAM
out_buffer_outermost:
	for (int l_block = 0; l_block < p_cBlocks; ++l_block) {
	    // Clear the buffer
	    ClearBuffer(l_buffer);
	    // Accumulate the blocks into the output submatrices
out_buffer_inner1:
	    for (int m = 0; m < p_aColBlocks; ++m) {
		BufferBlockStrassen_2(p_istream, l_buffer[0][0], 1, l_buffer[1][1], 1);
		BufferBlockStrassen_2(p_istream, l_buffer[1][0], 1, l_buffer[1][1], 0);
		BufferBlockStrassen_2(p_istream, l_buffer[0][1], 1, l_buffer[1][1], 1);
		BufferBlockStrassen_2(p_istream, l_buffer[0][0], 1, l_buffer[1][0], 1);
		BufferBlockStrassen_2(p_istream, l_buffer[0][0], 0, l_buffer[0][1], 1);
		BufferBlockStrassen_1(p_istream, l_buffer[1][1], 1);
		BufferBlockStrassen_1(p_istream, l_buffer[0][0], 1);
	    }
	    
	    // Stream out the buffer
out_buffer_inner2:
	    for (int i = 0; i < t_StrassensFactor; ++i) {
	            for (int l = 0; l < t_aColMemWords; ++l) {
			for (int p = 0; p < t_MemWidth; ++p) {
		for (int j = 0; j < t_StrassensFactor; ++j) {
#pragma HLS PIPELINE II = t_aColMemWords
	    	            for (int k = 0; k < t_aRowMemWords; ++k) {
				unsigned int src_offset = l * t_aMH + 
							  k * t_MemWidth + p;
				MemIntType l_val = l_buffer[i][j][src_offset];
		    	        p_ostream.write(l_val);
			    }
			}
		    }
	        }
	    }
	}
    }


    void StrassensWriteC(MemIntType* l_cAddr, 
		    	 MemStream& l_Cs, 
			 const unsigned int l_cRowBlocks, 
			 const unsigned int l_cColBlocks, 
			 const unsigned int l_cLd) {
	for (unsigned int l_cRowBlock = 0; l_cRowBlock < l_cRowBlocks; l_cRowBlock++) {
	    for (unsigned int l_cColBlock = 0; l_cColBlock < l_cColBlocks; l_cColBlock++) {
		t_subMatOps.WriteOutputStrassen(l_cAddr, 
						l_cRowBlock, 
						l_cColBlock, 
						l_cLd, 
						l_Cs);
	    }
	}
    }

    void GemmMicroKernel(MemStream& p_As,
		    	 MemStream& p_Bs,
			 MemStream& p_Cs,
			 unsigned int numBlocks) {
	const unsigned int p_transpBlocks = numBlocks * t_aRowMemWords;
	const unsigned int gemm_numBlocks = numBlocks * t_aRowMemWords * t_bColMemWords;
#pragma HLS DATAFLOW
	MemStream p_Bs1, p_AoutS;

	Transpose<t_FloatType, t_aColMemWords, t_MemWidth> l_transp(p_transpBlocks, t_bColMemWords);
	l_transp.process(p_As, p_AoutS);

	MatrixBuffer<typename MemWideType::t_TypeInt, t_MemWidth * t_aColMemWords, t_bColMemWords, true, false>()
	   .process(p_Bs, p_Bs1, numBlocks, t_aRowMemWords);

	Gemm<t_FloatType, t_bKD, t_MemWidth>::gemm(p_AoutS, p_Bs1, p_Cs, gemm_numBlocks);
    }
    
    void GemmBlocks(MemIntType* p_aAddr,
                    MemIntType* p_bAddr,
                    MemIntType* p_cAddr,
                    const unsigned int p_aColBlocks,
                    const unsigned int p_aRowBlocks,
                    const unsigned int p_bColBlocks,
                    const unsigned int p_aLd,
                    const unsigned int p_bLd,
                    const unsigned int p_cLd) {
        const unsigned int l_cBlocks = p_aRowBlocks * p_bColBlocks;
	const unsigned int l_abBlocks = l_cBlocks * p_aColBlocks;
	const unsigned int num_blocks_to_multiply = 7 * l_abBlocks;
#pragma HLS DATAFLOW
	MemStream l_Cs;
#pragma HLS STREAM variable = l_Cs depth = t_MemWidth * t_aRowMemWords * t_bColMemWords
#pragma HLS bind_storage variable = l_Cs type = fifo impl = uram

	MemStream l_res;
#pragma HLS STREAM variable = l_res depth = t_MemWidth * t_aRowMemWords * t_bColMemWords
#pragma HLS bind_storage variable = l_res type = fifo impl = uram

	MemStream l_lhs, l_rhs;
#pragma HLS STREAM variable = l_lhs depth = t_aColMemWords * t_MemWidth * t_aRowMemWords + 2
#pragma HLS bind_storage variable = l_lhs type = fifo impl = uram
#pragma HLS STREAM variable = l_rhs depth = t_aColMemWords * t_MemWidth * t_aRowMemWords + 2
#pragma HLS bind_storage variable = l_rhs type = fifo impl = uram

	GemmReadAB(p_aAddr, p_bAddr, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_aLd, p_bLd, l_lhs, l_rhs);

	GemmMicroKernel(l_lhs, l_rhs, l_res, num_blocks_to_multiply);

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
        MemIntType* l_cAddr = p_DdrWr + p_Args.m_Coffset * MemWideType::per4k();

        const unsigned int l_aColBlocks = p_Args.m_K / (t_MemWidth * t_aColMemWords * t_StrassensFactor);  // Number of blocks along matrix A cols
        const unsigned int l_aRowBlocks = p_Args.m_M / (t_MemWidth * t_aRowMemWords * t_StrassensFactor);
        const unsigned int l_bColBlocks = p_Args.m_N / (t_MemWidth * t_bColMemWords * t_StrassensFactor);
        const unsigned int l_aLd = p_Args.m_Lda / t_MemWidth;  // Number of Mem words in leading dim of A
        const unsigned int l_bLd = p_Args.m_Ldb / t_MemWidth;
        const unsigned int l_cLd = p_Args.m_Ldc / t_MemWidth;
        GemmBlocks(l_aAddr, l_bAddr, l_cAddr, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd);
    }
};
}
} // namespace
#endif
