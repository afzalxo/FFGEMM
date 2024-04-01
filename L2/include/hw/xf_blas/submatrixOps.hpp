#ifndef XF_BLAS_SUBMATRIXOPS_HPP
#define XF_BLAS_SUBMATRIXOPS_HPP

#include <cassert>
#include <iostream>
#include "types.hpp"
#include "gemm.hpp"

namespace xf {

namespace blas {

/**
 * @brief SubMatrixOps class defines the operations on submatrices given full matrix as input
 * Blocks are submatrices of the submatrices. Each block is of size 
 * (t_MemWidth * t_RowMemWords) x (t_MemWidth * t_ColMemWords) elements
 * while each submatrix is of size numRows / factor x numCols / factor elements
 * @param t_FloatType The data type of the matrix elements
 * @param t_MemWidth The number of matrix elements in one memory word
 * @param t_RowMemWords The number of memory words in one column of submatrix buffer
 * @param t_ColMemWords The number of memory words in one row of submatrix buffer
 */
template <typename t_FloatType,
	  unsigned int t_MemWidth,
	  unsigned int t_RowMemWords,  // Number of memory words in one column of submatrix buffer
	  unsigned int t_ColMemWords  // Number of memory words in one row of submatrix buffer
	  >
class SubMatrixOps {
    public:
      typedef WideType<t_FloatType, t_MemWidth> MemWideType;
      typedef typename MemWideType::t_TypeInt MemIntType;
      typedef hls::stream<MemIntType> MemStream;
      
      typedef hls::stream<typename TaggedWideType<t_FloatType, t_MemWidth>::t_TypeInt> EdgeStream;
      static const unsigned int t_aMH = t_MemWidth * t_RowMemWords;
      static const unsigned int t_2aMH = 2 * t_aMH;
      static const unsigned int t_2ColMemWords = 2 * t_ColMemWords;

    /* Read block of matrix from memory into p_BlockStream. The matrix is of size l_numElements x (l_numMemWords*t_MemWidth)
    // The block is located at row l_rowInd and col l_colInd in the matrix. The matrix is stored in row-major format.
    // l_wordLd is the number of mem words in the leading dimension of the matrix. E.g., for matrix of size 64 and 
    // memWidth of 16, l_wordLd would be 4 mem words.
    // Arguments: 
    // 		l_Addr: base address of the full matrix
    // 		l_rowInd: row index of the block
    // 		Note: Matrix is divided into blocks of size l_numElements x (l_numMemWords*t_MemWidth)
    // 		      The block is located at row l_rowInd and col l_colInd in the matrix. The matrix is stored in row-major format.
    // 		l_numElements: number of matrix elements in one col of matrix buffer
    // 		l_numMemWords: number of memory words in one row of matrix buffer
    // 		l_wordLd: number of mem words in the leading dimension of matrix. Currently row-major format.
    // 		p_BlockStream: output stream for the block
    */
      void ReadBlock(   MemIntType* l_Addr,
		    	const unsigned int l_rowInd,
			const unsigned int l_colInd,
			const unsigned int l_wordLd,
			MemStream& p_BlockStream    ) {
read_block_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = t_ColMemWords
read_block_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
	        unsigned int l_SrcOffset = 
		    l_wordLd * t_aMH * l_rowInd + l_colInd * t_ColMemWords + i * l_wordLd + j;
		MemIntType l_word = l_Addr[l_SrcOffset];
		p_BlockStream.write(l_word);
	    }
	}
      }

      void ReadAndBuffer(   MemIntType* l_Addr,
		      	   const unsigned int l_rowInd,
			   const unsigned int l_colInd,
			   const unsigned int l_wordLd,
			   MemWideType l_buf[t_aMH][t_ColMemWords]) {
read_to_buffer_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = t_ColMemWords
read_to_buffer_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
	        unsigned int l_SrcOffset = l_wordLd * t_aMH * l_rowInd + l_colInd * t_ColMemWords + i * l_wordLd + j;
		MemIntType l_word = l_Addr[l_SrcOffset];
		l_buf[i][j] = l_word;
	    }
	}
      }

      void ReadToBuffer_4x4_multidim(MemIntType* l_Addr,
		      		 const unsigned int l_rowInd,
				 const unsigned int l_colInd,
				 const unsigned int l_wordLd,
		      		 MemWideType l_buf[4][4][t_aMH][t_ColMemWords]) {
read_and_buffer_complete_outer:
          for (unsigned int i = 0; i < 4; ++i) {
              for (unsigned int k = 0; k < t_aMH; ++k) {
                  for (unsigned int j = 0; j < 4; ++j) {
                      for (unsigned int l = 0; l < t_ColMemWords; ++l) {
#pragma HLS PIPELINE II = 1
		          unsigned int src_idx = l_rowInd * 4 * t_aMH * l_wordLd + 
				  		 l_colInd * 4 * t_ColMemWords + 
						 i * (t_aMH * l_wordLd) + 
						 (j * t_ColMemWords) + 
						 k * l_wordLd + 
						 l;
		          MemWideType l_word = l_Addr[src_idx];
		          l_buf[i][j][k][l] = l_word;
		      }
		  }
	      }
	  }
      }

      void ReadToBuffer_4x4(     MemIntType* l_Addr,
		      		 const unsigned int l_rowInd,
				 const unsigned int l_colInd,
				 const unsigned int l_wordLd,
		      		 MemWideType l_buf[4*4*t_aMH*t_ColMemWords]) {
read_and_buffer_complete_outer:
	  for (unsigned int i = 0; i < 4; ++i) {
	      for (unsigned int k = 0; k < t_aMH; ++k) {
	          for (unsigned int j = 0; j < 4; ++j) {
	              for (unsigned int l = 0; l < t_ColMemWords; ++l) {
#pragma HLS PIPELINE II = 1
			  unsigned int src_idx = l_rowInd * 4 * t_aMH * l_wordLd + 
				  		 l_colInd * 4 * t_ColMemWords + 
						 i * (t_aMH * l_wordLd) + 
						 (j * t_ColMemWords) + 
						 k * l_wordLd + 
						 l;
			  unsigned int dst_idx = i * 4 * t_aMH * t_ColMemWords + 
				  		 j * t_ColMemWords +
						 k * 4 * t_ColMemWords +
						 l;
			  MemWideType l_word = l_Addr[src_idx];
			  l_buf[dst_idx] = l_word;
		      }
		  }
	      }
	  }
      }

      void AddBlocks_2(MemIntType* l_Addr,
		  	   MemWideType l_buf[4*4*t_aMH*t_ColMemWords],
			   const unsigned int l_buf0_blk_row,
			   const unsigned int l_buf0_blk_col,
			   const bool l_buf0_sign,
			   const unsigned int l_buf1_blk_row,
			   const unsigned int l_buf1_blk_col,
			   const bool l_buf1_sign,
			   MemStream& p_BlockStream) {
read_add_buffer_2_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = 2 * t_ColMemWords
read_add_buffer_2_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		MemWideType l_word(0);
		for (int k = 0; k < t_MemWidth; ++k) {
		    unsigned int l_srcOffset0 = l_buf0_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf0_blk_col * t_ColMemWords + 
						i * 4 * t_ColMemWords + 
						j;
		    unsigned int l_srcOffset1 = l_buf1_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf1_blk_col * t_ColMemWords +
						i * 4 * t_ColMemWords +
						j;
		    if (l_buf0_sign) {
		       l_word[k] += l_buf[l_srcOffset0][k];
		    } else {
		       l_word[k] -= l_buf[l_srcOffset0][k];
		    }
		    if (l_buf1_sign) {
		       l_word[k] += l_buf[l_srcOffset1][k];
		    } else {
		       l_word[k] -= l_buf[l_srcOffset1][k];
		    }
		}
		MemIntType l_word_int = l_word;
		p_BlockStream.write(l_word_int);
	    }
	}
      }

      void AddBlocks_3(    MemIntType* l_Addr,
		  	   MemWideType l_buf[4*4*t_aMH*t_ColMemWords],
			   const unsigned int l_buf0_blk_row,
			   const unsigned int l_buf0_blk_col,
			   const bool l_buf0_sign,
			   const unsigned int l_buf1_blk_row,
			   const unsigned int l_buf1_blk_col,
			   const bool l_buf1_sign,
			   const unsigned int l_buf2_blk_row,
			   const unsigned int l_buf2_blk_col,
			   const bool l_buf2_sign,
			   MemStream& p_BlockStream) {
read_add_buffer_3_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = 3 * t_ColMemWords
read_add_buffer_3_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		MemWideType l_word(0);
		for (int k = 0; k < t_MemWidth; ++k) {
		    unsigned int l_srcOffset0 = l_buf0_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf0_blk_col * t_ColMemWords + 
						i * 4 * t_ColMemWords + 
						j;
		    unsigned int l_srcOffset1 = l_buf1_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf1_blk_col * t_ColMemWords +
						i * 4 * t_ColMemWords +
						j;
		    unsigned int l_srcOffset2 = l_buf2_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf2_blk_col * t_ColMemWords +
						i * 4 * t_ColMemWords +
						j;
		    if (l_buf0_sign) {
		       l_word[k] += l_buf[l_srcOffset0][k];
		    } else {
		       l_word[k] -= l_buf[l_srcOffset0][k];
		    }
		    if (l_buf1_sign) {
		       l_word[k] += l_buf[l_srcOffset1][k];
		    } else {
		       l_word[k] -= l_buf[l_srcOffset1][k];
		    }
		    if (l_buf2_sign) {
		       l_word[k] += l_buf[l_srcOffset2][k];
		    } else {
		       l_word[k] -= l_buf[l_srcOffset2][k];
		    }
		}
		MemIntType l_word_int = l_word;
		p_BlockStream.write(l_word_int);
	    }
	}
      }

      void AddBlocks_4(    MemIntType* l_Addr,
		  	   MemWideType l_buf[4*4*t_aMH*t_ColMemWords],
			   const unsigned int l_buf0_blk_row,
			   const unsigned int l_buf0_blk_col,
			   const bool l_buf0_sign,
			   const unsigned int l_buf1_blk_row,
			   const unsigned int l_buf1_blk_col,
			   const bool l_buf1_sign,
			   const unsigned int l_buf2_blk_row,
			   const unsigned int l_buf2_blk_col,
			   const bool l_buf2_sign,
			   const unsigned int l_buf3_blk_row,
			   const unsigned int l_buf3_blk_col,
			   const bool l_buf3_sign,
			   MemStream& p_BlockStream) {
addblocks_4_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = 3 * t_ColMemWords
addblocks_4_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		MemWideType l_word(0);
		for (int k = 0; k < t_MemWidth; ++k) {
		    unsigned int l_srcOffset0 = l_buf0_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf0_blk_col * t_ColMemWords + 
						i * 4 * t_ColMemWords + 
						j;
		    unsigned int l_srcOffset1 = l_buf1_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf1_blk_col * t_ColMemWords +
						i * 4 * t_ColMemWords +
						j;
		    unsigned int l_srcOffset2 = l_buf2_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf2_blk_col * t_ColMemWords +
						i * 4 * t_ColMemWords +
						j;
		    unsigned int l_srcOffset3 = l_buf3_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf3_blk_col * t_ColMemWords +
						i * 4 * t_ColMemWords +
						j;
		    if (l_buf0_sign) {
		       l_word[k] += l_buf[l_srcOffset0][k];
		    } else {
		       l_word[k] -= l_buf[l_srcOffset0][k];
		    }
		    if (l_buf1_sign) {
		       l_word[k] += l_buf[l_srcOffset1][k];
		    } else {
		       l_word[k] -= l_buf[l_srcOffset1][k];
		    }
		    if (l_buf2_sign) {
		       l_word[k] += l_buf[l_srcOffset2][k];
		    } else {
		       l_word[k] -= l_buf[l_srcOffset2][k];
		    }
		    if (l_buf3_sign) {
		       l_word[k] += l_buf[l_srcOffset3][k];
		    } else {
		       l_word[k] -= l_buf[l_srcOffset3][k];
		    }
		}
		MemIntType l_word_int = l_word;
		p_BlockStream.write(l_word_int);
	    }
	}
      }

      void BufferToStream(MemWideType l_buf[4*4*t_aMH*t_ColMemWords],
		      	  const unsigned int l_blk_row,
			  const unsigned int l_blk_col,
		       	  MemStream& p_BlockStream) {
get_factor_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = t_ColMemWords
get_factor_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		unsigned int src_idx = l_blk_row * 4 * t_aMH * t_ColMemWords + 
				       l_blk_col * t_ColMemWords + 
				       i * 4 * t_ColMemWords + 
				       j;
		MemIntType l_word = l_buf[src_idx];
		p_BlockStream.write(l_word);
	    }
	}
      }

      void ReadAddSubBlock(MemIntType* l_Addr,
		      	   const unsigned int l_rowInd0,
			   const unsigned int l_colInd0,
			   const unsigned int l_rowInd1,
			   const unsigned int l_colInd1,
			   const unsigned int l_wordLd,
			   const unsigned int add_or_sub,
			   MemStream& p_BlockStream) {
read_add_sub_block_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = 2 * t_ColMemWords
read_add_sub_block_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		unsigned int l_SrcOffset_blk0 = l_wordLd * t_aMH * l_rowInd0 + l_colInd0 * t_ColMemWords + i * l_wordLd + j;
		unsigned int l_SrcOffset_blk1 = l_wordLd * t_aMH * l_rowInd1 + l_colInd1 * t_ColMemWords + i * l_wordLd + j;
	        MemWideType l_word0 = l_Addr[l_SrcOffset_blk0];
		MemWideType l_word1 = l_Addr[l_SrcOffset_blk1];
		MemWideType l_word;
		for (int k = 0; k < t_MemWidth; ++k) {
		    l_word[k] = l_word0[k] + l_word1[k];
		}
		p_BlockStream.write(l_word);
	    }
	}
      }

      void WriteOutputStrassen( MemIntType* l_Addr,
		       const unsigned int l_rowInd,
		       const unsigned int l_colInd,
		       const unsigned int l_wordLd,
		       MemStream& p_BlockStream) {
	for (int i = 0; i < 4; ++i) {
write_output_loop_tamh:
	    for (int k = 0; k < t_aMH; ++k) {
	        for (int j = 0; j < 4; ++j) {
#pragma HLS PIPELINE II = t_ColMemWords
		    for (int l = 0; l < t_ColMemWords; ++l) {
		        // The data in the p_BlockStream comes block by block, there are a total of 4x4 blocks,
		        // each of size t_aMH x t_ColMemWords. The data is written to the external memory at the appropriate location
		        // The data is written in row-major format.
		        unsigned int l_DstOffset = l_wordLd * (4 * t_aMH) * l_rowInd + l_colInd * (4 * t_ColMemWords) + (i * t_aMH * l_wordLd) + (j * t_ColMemWords) + k * l_wordLd + l;
			MemIntType l_word = p_BlockStream.read();
			MemWideType l_word_wide = l_word;
			l_Addr[l_DstOffset] = l_word;
		    }
		}
	    }
	}
      }

      void WriteOutputStrassen_nonsequential( MemIntType* l_Addr,
		       const unsigned int l_rowInd,
		       const unsigned int l_colInd,
		       const unsigned int l_wordLd,
		       MemStream& p_BlockStream) {
	for (int i = 0; i < 4; ++i) {
	    for (int j = 0; j < 4; ++j) {
		for (int p = 0; p < t_ColMemWords; ++p) {
write_output_loop_colmemwords:
		    for (int l = 0; l < t_ColMemWords; ++l) {
#pragma HLS PIPELINE II = t_MemWidth
			for (int k = 0; k < t_MemWidth; ++k) {
		        // The data in the p_BlockStream comes block by block, there are a total of 4x4 blocks,
		        // each of size t_aMH x t_ColMemWords. The data is written to the external memory at the appropriate location
		        // The data is written in row-major format.
			unsigned int l_DstOffset = l_wordLd * (4 * t_aMH) * l_rowInd + l_colInd * (4 * t_ColMemWords) + (i * t_aMH * l_wordLd) + (j * t_ColMemWords) + (k + p * t_MemWidth) * l_wordLd + l; // p * l_wordLd + l;
			MemIntType l_word = p_BlockStream.read();
			MemWideType l_word_wide = l_word;
			l_Addr[l_DstOffset] = l_word;
			}
		    }
		}
	    }
	}
      }

      // Splits the input stream into two output streams
      // Takes l_MemWords memory words from the input stream and writes them
      // to p_OutStream1
      void SplitStream(MemStream& p_InStream,
		       MemStream& p_OutStream1,
		       MemStream& p_OutStream2) {
split_stream_loop1:
	  for (unsigned int i = 0; i < t_aMH * t_ColMemWords; ++i) {
//#pragma HLS PIPELINE
	      MemWideType l_word = p_InStream.read();
	      p_OutStream1.write(l_word);
	  }
split_stream_loop2:
	  for (unsigned int i = 0; i < t_aMH * t_ColMemWords; ++i) {
//#pragma HLS PIPELINE
	      MemWideType l_word = p_InStream.read();
	      p_OutStream2.write(l_word);
	  }
      }

      void ExtractStream(MemStream& p_InStream,
		       MemStream& p_OutStream) {
extract_stream_loop:
	  for (int i = 0; i < t_aMH * t_ColMemWords; ++i) {
// #pragma HLS PIPELINE
	      p_OutStream.write(p_InStream.read());
	  }
      }

      /**
       * @brief WriteMemStream write matrix data to Memory
       *
       * @param l_cAddr  the base address of matrix in external memory
       * @param p_Cs  the input stream
       * @param l_aRowBlocks  the No. blocks along matrix X rows
       * @param l_bColBlocks  the No. blocks along matrix X cols
       * @param l_cWordLd  the matrix word leading dimention
       *
       */
      void WriteMemStream(MemIntType* l_cAddr,
                          MemStream& p_Cs,
                          unsigned int l_aRowBlocks,
                          unsigned int l_bColBlocks,
                          unsigned int l_cWordLd) {
          unsigned int l_rowOffset = 0;
          unsigned int l_colOffset = 0;

          for (int rowBlock = 0; rowBlock < l_aRowBlocks; ++rowBlock) {
              for (int colBlock = 0; colBlock < l_bColBlocks; ++colBlock) {
                  for (int i = 0; i < t_RowMemWords * t_MemWidth; ++i)
#pragma HLS PIPELINE II = t_ColMemWords
                      for (int j = 0; j < t_ColMemWords; j++) {
                          unsigned int l_dstOffset = i * l_cWordLd + l_cWordLd * t_MemWidth * t_RowMemWords * rowBlock +
                                                     colBlock * t_ColMemWords;
                          MemIntType l_val = p_Cs.read();
                          l_cAddr[l_dstOffset + j] = l_val;
                      }
              }
          }
      }

      void WriteStreamToMemDebug(MemIntType* l_cAddr,
          	    	       MemStream& p_Cs,
          		       unsigned int l_numMemWords) {
          unsigned int l_dstOffset = 0;
          for (int i = 0; i < l_numMemWords; ++i) {
              MemIntType l_val = p_Cs.read();
              l_cAddr[l_dstOffset + i] = l_val;
          }
      }


};

}

}

#endif // XF_BLAS_SUBMATRIXOPS_HPP
