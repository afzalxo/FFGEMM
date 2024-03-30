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
		// hls::print("l_SrcOffset = %d\n", l_SrcOffset);
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

      void ReadAndBufferComplete(MemIntType* l_Addr,
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

      void ReadAndBufferComplete_flattened(MemIntType* l_Addr,
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

      void ReadAddBuffer_2(MemIntType* l_Addr,
		  	   MemWideType l_buf0[t_aMH][t_ColMemWords],
			   const bool l_buf0_sign,
			   MemWideType l_buf1[t_aMH][t_ColMemWords],
			   const bool l_buf1_sign,
			   MemStream& p_BlockStream) { 
read_add_buffer_2_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = 2 * t_ColMemWords
read_add_buffer_2_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		MemWideType l_word(0);
		for (int k = 0; k < t_MemWidth; ++k) {
		    // l_word[k] = l_buf0[i][j][k] + l_buf1[i][j][k];
		    if (l_buf0_sign) {
		       l_word[k] += l_buf0[i][j][k];
		    } else {
		       l_word[k] -= l_buf0[i][j][k];
		    }
		    if (l_buf1_sign) {
		       l_word[k] += l_buf1[i][j][k];
		    } else {
		       l_word[k] -= l_buf1[i][j][k];
		    }
		}
		MemIntType l_word_int = l_word;
		p_BlockStream.write(l_word_int);
	    }
	}
      }

      void ReadAddBuffer_2_flattened(MemIntType* l_Addr,
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
						j;// * t_MemWidth;
		    unsigned int l_srcOffset1 = l_buf1_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf1_blk_col * t_ColMemWords +
						i * 4 * t_ColMemWords +
						j;// * t_MemWidth;
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


      void ReadAddBuffer_3(MemIntType* l_Addr,
		  	   MemWideType l_buf0[t_aMH][t_ColMemWords],
			   const bool l_buf0_sign,
			   MemWideType l_buf1[t_aMH][t_ColMemWords],
			   const bool l_buf1_sign,
			   MemWideType l_buf2[t_aMH][t_ColMemWords],
			   const bool l_buf2_sign,
			   MemStream& p_BlockStream) { 
read_add_buffer_3_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = 3 * t_ColMemWords
read_add_buffer_3_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		MemWideType l_word(0);
		for (int k = 0; k < t_MemWidth; ++k) {
		    // l_word[k] = l_buf0[i][j][k] + l_buf1[i][j][k] + l_buf2[i][j][k];
		    if (l_buf0_sign) {
		       l_word[k] += l_buf0[i][j][k];
		    } else {
		       l_word[k] -= l_buf0[i][j][k];
		    }
		    if (l_buf1_sign) {
		       l_word[k] += l_buf1[i][j][k];
		    } else {
		       l_word[k] -= l_buf1[i][j][k];
		    }
		    if (l_buf2_sign) {
		       l_word[k] += l_buf2[i][j][k];
		    } else {
		       l_word[k] -= l_buf2[i][j][k];
		    }
		}
		MemIntType l_word_int = l_word;
		p_BlockStream.write(l_word_int);
	    }
	}
      }

      void ReadAddBuffer_3_flattened(MemIntType* l_Addr,
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
						j;// * t_MemWidth;
		    unsigned int l_srcOffset1 = l_buf1_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf1_blk_col * t_ColMemWords +
						i * 4 * t_ColMemWords +
						j;// * t_MemWidth;
		    unsigned int l_srcOffset2 = l_buf2_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf2_blk_col * t_ColMemWords +
						i * 4 * t_ColMemWords +
						j;// * t_MemWidth;
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

      void ReadAddBuffer_4(MemIntType* l_Addr,
		  	   MemWideType l_buf0[t_aMH][t_ColMemWords],
			   const bool l_buf0_sign,
			   MemWideType l_buf1[t_aMH][t_ColMemWords],
			   const bool l_buf1_sign,
			   MemWideType l_buf2[t_aMH][t_ColMemWords],
			   const bool l_buf2_sign,
			   MemWideType l_buf3[t_aMH][t_ColMemWords],
			   const bool l_buf3_sign,
			   MemStream& p_BlockStream) { 
read_add_buffer_4_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = 4 * t_ColMemWords
read_add_buffer_4_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		MemWideType l_word(0);
		for (int k = 0; k < t_MemWidth; ++k) {
		    // l_word[k] = l_buf0[i][j][k] + l_buf1[i][j][k] + l_buf2[i][j][k] + l_buf3[i][j][k];
		    if (l_buf0_sign) {
		       l_word[k] += l_buf0[i][j][k];
		    } else {
		       l_word[k] -= l_buf0[i][j][k];
		    }
		    if (l_buf1_sign) {
		       l_word[k] += l_buf1[i][j][k];
		    } else {
		       l_word[k] -= l_buf1[i][j][k];
		    }
		    if (l_buf2_sign) {
		       l_word[k] += l_buf2[i][j][k];
		    } else {
		       l_word[k] -= l_buf2[i][j][k];
		    }
		    if (l_buf3_sign) {
		       l_word[k] += l_buf3[i][j][k];
		    } else {
		       l_word[k] -= l_buf3[i][j][k];
		    }
		}
		MemIntType l_word_int = l_word;
		p_BlockStream.write(l_word_int);
	    }
	}
      }

      void ReadAddBuffer_4_flattened(MemIntType* l_Addr,
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
						j;// * t_MemWidth;
		    unsigned int l_srcOffset1 = l_buf1_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf1_blk_col * t_ColMemWords +
						i * 4 * t_ColMemWords +
						j;// * t_MemWidth;
		    unsigned int l_srcOffset2 = l_buf2_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf2_blk_col * t_ColMemWords +
						i * 4 * t_ColMemWords +
						j;// * t_MemWidth;
		    unsigned int l_srcOffset3 = l_buf3_blk_row * 4 * t_aMH * t_ColMemWords + 
			    			l_buf3_blk_col * t_ColMemWords +
						i * 4 * t_ColMemWords +
						j;// * t_MemWidth;
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

      void AddFactors(  MemWideType l_buf0[t_aMH][t_ColMemWords],
		       MemWideType l_buf1[t_aMH][t_ColMemWords],
		       MemStream& p_BlockStream) {
add_factors_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = t_ColMemWords
add_factors_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		MemWideType l_word;
		for (int k = 0; k < t_MemWidth; ++k) {
		    l_word[k] = l_buf0[i][j][k] + l_buf1[i][j][k];
		}
		p_BlockStream.write(l_word);
	    }
	}
      }

      void BlockBufferToStream(MemWideType l_buf[t_aMH][t_ColMemWords],
		       	       MemStream& p_BlockStream) {
get_factor_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = t_ColMemWords
get_factor_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		MemIntType l_word = l_buf[i][j];
		p_BlockStream.write(l_word);
	    }
	}
      }

      void BlockBufferToStream_flattened(MemWideType l_buf[4*4*t_aMH*t_ColMemWords],
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

      void ReadAddPopulateBuffers(MemIntType* l_Addr,
		      		  const unsigned int l_rowInd0,
				  const unsigned int l_colInd0,
				  const unsigned int l_rowInd1,
				  const unsigned int l_colInd1,
				  const unsigned int l_wordLd,
				  MemWideType l_buf0[t_aMH][t_ColMemWords],
				  MemWideType l_buf1[t_aMH][t_ColMemWords],
				  MemStream& p_BlockStream) { 
read_add_populate_buffers_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = 2 * t_ColMemWords
read_add_populate_buffers_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		unsigned int l_SrcOffset_blk0 = l_wordLd * t_aMH * l_rowInd0 + l_colInd0 * t_ColMemWords + i * l_wordLd + j;
		unsigned int l_SrcOffset_blk1 = l_wordLd * t_aMH * l_rowInd1 + l_colInd1 * t_ColMemWords + i * l_wordLd + j;
	        MemWideType l_word0 = l_Addr[l_SrcOffset_blk0];
		MemWideType l_word1 = l_Addr[l_SrcOffset_blk1];
		l_buf0[i][j] = l_word0;
		l_buf1[i][j] = l_word1;
		MemWideType l_word;
		for (int k = 0; k < t_MemWidth; ++k) {
		    l_word[k] = l_word0[k] + l_word1[k];
		}
		p_BlockStream.write(l_word);
	    }
	}
      }

      void ReadAddToBuffer(MemIntType* l_Addr,
		      	   const unsigned int l_rowInd,
			   const unsigned int l_colInd,
			   const unsigned int l_wordLd,
			   MemWideType l_buf_rd[t_aMH][t_ColMemWords],
			   MemWideType l_buf_wr[t_aMH][t_ColMemWords],
			   MemStream& p_BlockStream) {
read_add_to_buffer_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = t_ColMemWords
read_add_to_buffer_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		unsigned int l_SrcOffset = l_wordLd * t_aMH * l_rowInd + l_colInd * t_ColMemWords + i * l_wordLd + j;
		MemWideType l_word = l_Addr[l_SrcOffset];
		l_buf_wr[i][j] = l_word;
		MemWideType o_word;
		for (int k = 0; k < t_MemWidth; ++k) {
		    o_word[k] = l_word[k] + l_buf_rd[i][j][k];
		}
		p_BlockStream.write(o_word);
	    }
	}
      }

      void AddBuffers( MemWideType l_buf0[t_aMH][t_ColMemWords],
		       MemWideType l_buf1[t_aMH][t_ColMemWords],
		       MemStream& p_BlockStream) {
add_buffers_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = t_ColMemWords
add_buffers_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		MemWideType l_word;
		for (int k = 0; k < t_MemWidth; ++k) {
		    l_word[k] = l_buf0[i][j][k] + l_buf1[i][j][k];
		}
		p_BlockStream.write(l_word);
	    }
	}
      }

      void BufferToStream(MemWideType l_buf[t_aMH][t_ColMemWords],
		      	 MemStream& p_BlockStream) {
buffer_to_stream_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = t_ColMemWords
buffer_to_stream_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		p_BlockStream.write(l_buf[i][j]);
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

      void WriteOutputStrassen_New( MemIntType* l_Addr,
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



      // Adds two matrix blocks element-wise and writes the result to the output stream
      // The matrices are of sizes l_numElemsCol x (l_numMemWordsRow*t_MemWidth)
      void AddBlocksStreams(   MemStream& p_StreamA,
          	    	MemStream& p_StreamB,
          		MemStream& p_StreamC) {
add_blocks_outer:
          for (unsigned int r = 0; r < t_aMH; ++r) {
add_blocks_inner:
              for (unsigned int c = 0; c < t_ColMemWords; ++c) {
// #pragma HLS PIPELINE II = t_MemWidth
                  MemWideType l_wordA = p_StreamA.read();
          	  MemWideType l_wordB = p_StreamB.read();
          	  MemWideType l_wordC;
          	  for (unsigned int i = 0; i < t_MemWidth; ++i) {
          	      l_wordC[i] = l_wordA[i] + l_wordB[i];
          	  }
          	  p_StreamC.write(l_wordC);
              }
          }
      }

      void AddBlocks_2(MemWideType l_buf0[t_aMH][t_ColMemWords],
		       MemWideType l_buf1[t_aMH][t_ColMemWords],
		       MemStream& p_BlockStream) {
add_blocks_2_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = t_ColMemWords
add_blocks_2_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		MemWideType l_word;
		for (int k = 0; k < t_MemWidth; ++k) {
		    l_word[k] = l_buf0[i][j][k] + l_buf1[i][j][k];
		}
		p_BlockStream.write(l_word);
	    }
	}
      }

      void AddBlocks_3(MemWideType l_buf0[t_aMH][t_ColMemWords],
		       MemWideType l_buf1[t_aMH][t_ColMemWords],
		       MemWideType l_buf2[t_aMH][t_ColMemWords],
		       MemStream& p_BlockStream) {
add_blocks_3_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = t_ColMemWords
add_blocks_3_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		MemWideType l_word;
		for (int k = 0; k < t_MemWidth; ++k) {
		    l_word[k] = l_buf0[i][j][k] + l_buf1[i][j][k] + l_buf2[i][j][k];
		}
		p_BlockStream.write(l_word);
	    }
	}
      }

      void AddBlocks_4(MemWideType l_buf0[t_aMH][t_ColMemWords],
		       MemWideType l_buf1[t_aMH][t_ColMemWords],
		       MemWideType l_buf2[t_aMH][t_ColMemWords],
		       MemWideType l_buf3[t_aMH][t_ColMemWords],
		       MemStream& p_BlockStream) {
add_blocks_4_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE II = t_ColMemWords
add_blocks_4_inner:
	    for (int j = 0; j < t_ColMemWords; ++j) {
		MemWideType l_word;
		for (int k = 0; k < t_MemWidth; ++k) {
		    l_word[k] = l_buf0[i][j][k] + l_buf1[i][j][k] + l_buf2[i][j][k] + l_buf3[i][j][k];
		}
		p_BlockStream.write(l_word);
	    }
	}
      }

      void SubBlocks(   MemStream& p_StreamA,
          	    	MemStream& p_StreamB,
          		MemStream& p_StreamC) {
sub_blocks_outer:
          for (int r = 0; r < t_aMH; ++r) {
sub_blocks_inner:
              for (int c = 0; c < t_ColMemWords; ++c) {
// #pragma HLS PIPELINE II = t_MemWidth
                  MemWideType l_wordA = p_StreamA.read();
          	  MemWideType l_wordB = p_StreamB.read();
          	  MemWideType l_wordC;
          	  for (int i = 0; i < t_MemWidth; ++i) {
          	      l_wordC[i] = l_wordA[i] - l_wordB[i];
          	  }
          	  p_StreamC.write(l_wordC);
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

      /* @brief AddConsecutiveBlocks adds consecutive blocks of a matrix and writes the
       * result to the output stream. The blocks are of sizes (l_numElements x (l_numMemWords * t_MemWidth)) elements
       * @param p_InStream  the input stream contianing the two blocks contiguously streamed
       * @param p_OutStream  the output stream
       * @param l_numElements  the number of matrix elements along the rows of the block (i.e., height of the block)
       * @param l_numMemWords  the number of memory words along the columns of the block (i.e., width of the block in MemWords)
       */
      void AddConsecutiveBlocks(MemStream& p_InStream,
		      		MemStream& p_OutStream) {
	   MemStream l_Ins1, l_Ins2;
#pragma HLS STREAM variable=l_Ins1 depth=t_aMH * t_ColMemWords + 2// depth=l_numElements*l_numMemWords
#pragma HLS STREAM variable=l_Ins2 depth=2//l_numElements*l_numMemWords
#pragma HLS DATAFLOW
	   SplitStream(p_InStream, l_Ins1, l_Ins2);
	   AddBlocks(l_Ins1, l_Ins2, p_OutStream);
      }

      void SubConsecutiveBlocks(MemStream& p_InStream,
		      		MemStream& p_OutStream) {
	   MemStream l_Ins1, l_Ins2;
#pragma HLS STREAM variable=l_Ins1 depth=t_aMH * t_ColMemWords + 2//l_numElements*l_numMemWords
#pragma HLS STREAM variable=l_Ins2 depth=2
#pragma HLS DATAFLOW
// #pragma HLS STREAM variable=l_Ins2 depth=128//l_numElements*l_numMemWords
	   SplitStream(p_InStream, l_Ins1, l_Ins2);
	   SubBlocks(l_Ins1, l_Ins2, p_OutStream);
      }

      /* @brief AddSubmatrices adds two submatrices and writes the result to the output stream
       * The submatrices are of sizes TODO
       * @param p_InAddr  the base address of matrix in external memory
       * @param p_OutAddr  the base address of matrix in external memory
       * @param rowInd1  the row index of the first submatrix
       * @param colInd1  the column index of the first submatrix
       * @param rowInd2  the row index of the second submatrix
       * @param colInd2  the column index of the second submatrix
       * @param numRows  the number of rows in the matrix
       * @param numCols  the number of columns in the matrix
       */
      void AddSubmatrices(MemIntType* p_InAddr,
          	    	MemIntType* p_OutAddr,
          	    	unsigned int rowInd1,
          		unsigned int colInd1,
          		unsigned int rowInd2,
          		unsigned int colInd2,
          		unsigned int numRows,
          		unsigned int numCols) {
          const unsigned int l_submatColBlocks = numCols / (2 * (t_MemWidth * t_ColMemWords));
          const unsigned int l_submatRowBlocks = numRows / (2 * (t_MemWidth * t_RowMemWords));
          MemStream l_Ins1, l_Ins2, l_Outs;
          for (int l_submatRowBlock = 0; l_submatRowBlock < l_submatRowBlocks; ++l_submatRowBlock) {
              for (int l_submatColBlock = 0; l_submatColBlock < l_submatColBlocks; ++l_submatColBlock) {
          	ReadBlock(p_InAddr, 
          		      2 * rowInd1 + l_submatRowBlock, 
          		      2 * colInd1 + l_submatColBlock, 
          		      t_MemWidth * t_RowMemWords,
          		      t_ColMemWords, 
          		      numCols / t_MemWidth, 
          		      l_Ins1);
          	ReadBlock(p_InAddr, 
          		      2 * rowInd2 + l_submatRowBlock, 
          		      2 * colInd2 + l_submatColBlock, 
          		      t_MemWidth * t_RowMemWords,
          		      t_ColMemWords, 
          		      numCols / t_MemWidth, 
          		      l_Ins2);
          	AddBlocks(t_MemWidth * t_RowMemWords, t_ColMemWords, l_Ins1, l_Ins2, l_Outs);
              }
          }
          // WriteMemStream(p_OutAddr, l_Outs, l_submatRowBlocks, l_submatColBlocks, numCols / (t_MemWidth));
          WriteMemStream(p_OutAddr, l_Outs, 1, 1, numCols / (2 * t_MemWidth));
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


template <typename MemIntType, typename MemWideType, typename MemStream,
	  unsigned int t_MemWidth, unsigned int t_RowMemWords, unsigned int t_ColMemWords,
	  unsigned int t_aMH, unsigned int p_numBlocks>
void ReadAddBlocks(MemIntType* l_Addr,
		   const unsigned int l_rowInd,
		   const unsigned int l_colInd,
		   const unsigned int l_wordLd,
		   MemWideType l_buffer[2][2][t_aMH][t_ColMemWords],
		   ap_uint<1> l_buf_valid[2][2],
		   unsigned int* l_blk_idx_rows,
		   unsigned int* l_blk_idx_cols,
		   MemStream& p_BlockStream) {
read_add_blocks_outer:
	for (int i = 0; i < t_aMH; ++i) {
#pragma HLS PIPELINE
	    for (int j = 0; j < t_ColMemWords; ++j) {
		MemWideType l_words[p_numBlocks];
		for (int k = 0; k < p_numBlocks; ++k) {
		    unsigned int l_SrcOffset = l_wordLd * t_aMH * ((l_rowInd << 1) + l_blk_idx_rows[k]) + ((l_colInd << 1) + l_blk_idx_cols[k]) * t_ColMemWords + i * l_wordLd + j;
		    if (l_buf_valid[l_blk_idx_rows[k]][l_blk_idx_cols[k]]) {
		        l_words[k] = l_buffer[l_blk_idx_rows[k]][l_blk_idx_cols[k]][i][j];
		    } else {
		        l_words[k] = l_Addr[l_SrcOffset];
			l_buffer[l_blk_idx_rows[k]][l_blk_idx_cols[k]][i][j] = l_words[k];
		    }
		}
		MemWideType l_outWord;
		for (int k = 0; k < t_MemWidth; ++k) {
		    l_outWord[k] = 0;;
		    for (int l = 0; l < p_numBlocks; ++l) {
			l_outWord[k] += l_words[l][k];
		    }
		}
		p_BlockStream.write(l_outWord);
	    }
	}
	for (int k = 0; k < p_numBlocks; ++k) {
	    if (l_buf_valid[l_blk_idx_rows[k]][l_blk_idx_cols[k]] == 0) {
	        l_buf_valid[l_blk_idx_rows[k]][l_blk_idx_cols[k]] = 1;
	    }
	}
}

}

}

#endif // XF_BLAS_SUBMATRIXOPS_HPP
