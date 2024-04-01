#include "uut_top.hpp"


extern "C" {

void uut_top(   MemIntType* l_aAddr, 
		MemIntType* l_bAddr, 
		MemIntType* l_cAddr, 
		unsigned int l_aColBlocks, 
		unsigned int l_aRowBlocks,
		unsigned int l_bColBlocks, 
		unsigned int l_aLd, 
		unsigned int l_bLd,
		unsigned int l_cLd) {
#pragma HLS INTERFACE m_axi bundle = gmem0 depth = BLAS_m * (BLAS_k / BLAS_memWidth) port = l_aAddr
#pragma HLS INTERFACE m_axi bundle = gmem0 depth = BLAS_m * (BLAS_k / BLAS_memWidth) port = l_bAddr
#pragma HLS INTERFACE m_axi bundle = gmem0 depth = BLAS_m * (BLAS_k / BLAS_memWidth) port = l_cAddr

#if RUN_STRASSENS
  GemmType l_gemmKernel;
  l_gemmKernel.GemmBlocks(l_aAddr, l_bAddr, l_cAddr, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd);
#elif RUN_BASELINE
  GemmTypeBaseline l_gemmKernel;
  const unsigned int l_transpBlocks = l_aColBlocks * l_aRowBlocks * l_bColBlocks * BLAS_gemmMBlocks;
  l_gemmKernel.GemmBlocks(l_aAddr, l_bAddr, l_cAddr, l_cAddr, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd, l_cLd, l_transpBlocks, 1);
#else
  std::cout << "ERROR: No implementation selected" << std::endl;
#endif
}

void uut_top_test(  MemIntType* l_aAddr, 
		    MemIntType* l_bAddr, 
		    MemIntType* l_cAddr, 
		    unsigned int l_aColBlocks, 
		    unsigned int l_aRowBlocks,
		    unsigned int l_bColBlocks, 
		    unsigned int l_aLd, 
		    unsigned int l_bLd,
		    unsigned int l_cLd, 
		    unsigned int l_transpBlocks) {
#pragma HLS INTERFACE m_axi bundle = gmem0 depth = BLAS_m * (BLAS_k / BLAS_memWidth) port = l_aAddr
#pragma HLS INTERFACE m_axi bundle = gmem0 depth = BLAS_m * (BLAS_k / BLAS_memWidth) port = l_bAddr
#pragma HLS INTERFACE m_axi bundle = gmem0 depth = BLAS_m * (BLAS_k / BLAS_memWidth) port = l_cAddr
  MemStream s_lhs, s_rhs;

  GemmType l_gemmKernel;
  // l_gemmKernel.GemmReadAB_Auto(l_aAddr, l_bAddr, 1, 1, 1, l_aLd, l_bLd, s_lhs, s_rhs);
}
}
