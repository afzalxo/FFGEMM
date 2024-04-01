#include "params.hpp"
#include "types.hpp"
#include "uut_top.hpp"
#include <stdio.h>
#include <iostream>


void multiply_matrices_sw(BLAS_dataType* in1, BLAS_dataType* in2, BLAS_dataType* out, int dim) {
  for (int k = 0; k < dim; k++)
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++)
        out[i*dim + j] += in1[i * dim + k] * in2[k * dim + j];
}

void pack_matrix(BLAS_dataType* in, MemIntType* packed, int dim_in_r, int dim_in_c) {
  for (int i = 0; i < dim_in_r; i++) {
    for (int j = 0; j < dim_in_c; j+= BLAS_memWidth) {
      MemWideType tmp;
      for (int k = 0; k < BLAS_memWidth; k++) {
	tmp[k] = in[i*dim_in_c + j + k];
      }
      packed[i*dim_in_c/BLAS_memWidth + j/BLAS_memWidth] = tmp;
    }
  }
}

void unpack_matrix(MemIntType* packed, BLAS_dataType* unpacked, int dim_in_r, int dim_in_c) {
  for (int i = 0; i < dim_in_r; i++) {
    for (int j = 0; j < dim_in_c; j+= BLAS_memWidth) {
      unsigned int packed_idx = (i * dim_in_c) / BLAS_memWidth + j / BLAS_memWidth;
      MemWideType tmp = packed[packed_idx];
      for (int k = 0; k < BLAS_memWidth; k++) {
	unsigned int unpacked_idx = i * dim_in_c + j + k;
	unpacked[unpacked_idx] = tmp[k];
      }
    }
  }
}

void unpack_debug(MemIntType* packed, BLAS_dataType* unpacked, int dims) {
  for (int i = 0; i < dims; i++) {
    MemWideType tmp = packed[i];
    for (int j = 0; j < BLAS_memWidth; j++) {
      unpacked[i*BLAS_memWidth + j] = tmp[j];
    }
  }
}

void print_matrix(BLAS_dataType* matrix, int dim_r, int dim_c) {
  for (int i = 0; i < dim_r; i++) {
    for (int j = 0; j < dim_c; j++) {
      if (j == dim_c - 1) {
	std::cout << matrix[i*dim_c + j] << ";" << std::endl;
      } else {
        std::cout << matrix[i*dim_c + j] << ",";
      }
    }
    std::cout << std::endl;
  }
}

void print_submatrix(BLAS_dataType* matrix, int dim_r, int dim_c, int block_row_idx, int block_col_idx) {
  // Divide the matrix into blocks of size BLAS_m // 4 x BLAS_n // 4
  // and print the block at block_row_idx, block_col_idx
  int block_dim_r = dim_r / 4;
  int block_dim_c = dim_c / 4;
  for (int i = block_row_idx * block_dim_r; i < (block_row_idx + 1) * block_dim_r; i++) {
    for (int j = block_col_idx * block_dim_c; j < (block_col_idx + 1) * block_dim_c; j++) {
      if (j == (block_col_idx + 1) * block_dim_c - 1) {
	std::cout << matrix[i*dim_c + j] << ";" << std::endl;
      } else {
	std::cout << matrix[i*dim_c + j] << ",";
      }
    }
  }
}

int main() {
  int matrix_size = BLAS_m;

  const unsigned int t_aColMemWords = BLAS_gemmKBlocks;
  const unsigned int t_aRowMemWords = BLAS_gemmMBlocks;
  const unsigned int t_bColMemWords = BLAS_gemmNBlocks;

  typedef WideType<BLAS_dataType, BLAS_memWidth> MemWideType;
  typedef typename MemWideType::t_TypeInt MemIntType;

  const unsigned int memWordsaCol = BLAS_k / BLAS_memWidth;
  const unsigned int memWordsbCol = BLAS_n / BLAS_memWidth;

  MemIntType l_aAddr[BLAS_m * memWordsaCol];
  MemIntType l_bAddr[memWordsaCol * BLAS_n];
  MemIntType l_cAddr[BLAS_m * memWordsbCol];

  BLAS_dataType in1[matrix_size*matrix_size];
  BLAS_dataType in2[matrix_size*matrix_size];
  BLAS_dataType out_mat[matrix_size*matrix_size];
  BLAS_dataType golden_out[matrix_size*matrix_size];
  // int32_t debug_out[64*64];
  const int matrix_max = 64;
  const int matrix_min = 7;
  const int matrix_range = matrix_max - matrix_min + 1;

  for (int i = 0; i < matrix_size; i++) {
    for (int j = 0; j < matrix_size; j++) {
      in1[i*matrix_size + j] = std::rand() % matrix_range + matrix_min; //((i * matrix_size + j) % 13 + 1);
      in2[i*matrix_size + j] = std::rand() % matrix_range + matrix_min; //((i * matrix_size + j) % 13 + 1);
    }
  }

  for (int i = 0; i < matrix_size*matrix_size; i++) {
    golden_out[i] = 0;
    out_mat[i] = 0;
  }
  // for (int i = 0; i < 64 * 64; ++i) {
  //    debug_out[i] = 0;
  // }

  std::cout << "============================================" << std::endl;
  std::cout << "Gemm Test: " << std::endl;
  // std::cout << "BLAS_dataType = " << str(BLAS_dataType) << std::endl;
  std::cout << "M = " << BLAS_m << ", N = " << BLAS_n << ", K = " << BLAS_k << std::endl;
  std::cout << "MemWidth = " << BLAS_memWidth << ", BLAS_gemmMBlocks = " << BLAS_gemmMBlocks  << std::endl;
  std::cout << "============================================" << std::endl;

  std::cout << "Running Sw-based Gemm..." << std::endl;
  multiply_matrices_sw(in1, in2, golden_out, matrix_size);

  std::cout << "Packing matrices from int32_t to MemIntType..." << std::endl;
  pack_matrix(in1, l_aAddr, BLAS_m, BLAS_k);
  pack_matrix(in2, l_bAddr, BLAS_k, BLAS_n);

  for (int i = 0; i < BLAS_m * memWordsbCol; i++) {
    l_cAddr[i] = 0;
  }

#if RUN_STRASSENS
  const unsigned int l_aColBlocks = BLAS_k / (BLAS_memWidth * t_aColMemWords * 4);
  const unsigned int l_aRowBlocks = BLAS_m / (BLAS_memWidth * t_aRowMemWords * 4);
  const unsigned int l_bColBlocks = BLAS_n / (BLAS_memWidth * t_bColMemWords * 4);
#elif RUN_BASELINE
  const unsigned int l_aColBlocks = BLAS_k / (BLAS_memWidth * t_aColMemWords);
  const unsigned int l_aRowBlocks = BLAS_m / (BLAS_memWidth * t_aRowMemWords);
  const unsigned int l_bColBlocks = BLAS_n / (BLAS_memWidth * t_bColMemWords);
#else
  std::cout << "ERROR: No architecture defined" << std::endl;
  return 1;
#endif

  const unsigned int l_aLd = BLAS_k / BLAS_memWidth;
  const unsigned int l_bLd = BLAS_n / BLAS_memWidth;
  const unsigned int l_cLd = BLAS_n / BLAS_memWidth;

  // unsigned int l_transpBlocks = l_aColBlocks * l_aRowBlocks * l_bColBlocks * t_aRowMemWords;

  std::cout << "Running Gemm Sim using Kernel..." << std::endl;
  uut_top(l_aAddr, l_bAddr, l_cAddr, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd);
  // unpack_debug(l_cAddr, debug_out, 64 * 4);
  //for (int i = 0; i < 64 * 64; ++i) {
  //  std::cout << "debug_out[" << i << "] = " << debug_out[i] << std::endl;
  //}
  std::cout << "Unpacking matrices from MemIntType to int32_t..." << std::endl;
  unpack_matrix(l_cAddr, out_mat, BLAS_m, BLAS_n);

  // print_matrix(out_mat, BLAS_m, BLAS_n);
  // print_submatrix(out_mat, BLAS_m, BLAS_n, 0, 0);
  std::cout << "Comparing results..." << std::endl;
  bool pass_or_fail = 0;
  unsigned int error_count = 0;
  unsigned int num_pass = 0;
  for (int i = 0; i < matrix_size*matrix_size; i++) {
    if (golden_out[i] != out_mat[i]) {
      std::cout << "Test FAILED: Entry at index " << i 
	      << ", Golden: " << golden_out[i] 
	      << " != Kernel Output: " << out_mat[i] << std::endl;
      pass_or_fail = 1;
      error_count++;
    } else {
      num_pass++;
    }
  }
  if (pass_or_fail == 0) {
    std::cout << "Functional: Test PASSED!" << std::endl;
  } else {
    std::cout << "Functional: Test FAILED!, Total: " << matrix_size * matrix_size << ", Number Passed: " << num_pass << ", Number Failed: " << error_count  << std::endl;
  }
  return 0;
}
