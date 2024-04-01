#ifndef UUT_TOP_HPP
#define UUT_TOP_HPP

#include "mem.hpp"
#include "kargs.hpp"
#include "params.hpp"
#include "gemmKernel.hpp"
#include "strassensSquaredKernel.hpp"
#include "types.hpp"


#if RUN_STRASSENS
typedef xf::blas::StrassensSquaredKernel<BLAS_dataType,
				  	BLAS_memWidth,
				  	BLAS_gemmKBlocks,
				  	BLAS_gemmMBlocks,
				  	BLAS_gemmNBlocks> GemmType;
#elif RUN_BASELINE
typedef xf::blas:: GemmKernel<BLAS_dataType,
			      BLAS_dataType,
			      BLAS_memWidth,
			      BLAS_memWidth,
			      BLAS_gemmKBlocks,
			      BLAS_gemmMBlocks,
			      BLAS_gemmNBlocks> GemmTypeBaseline;
#endif

typedef WideType<BLAS_dataType, BLAS_memWidth> MemWideType;
typedef typename MemWideType::t_TypeInt MemIntType;
typedef hls::stream<MemIntType> MemStream;

extern "C" {
    void uut_top( MemIntType* l_aAddr, 
		  MemIntType* l_bAddr, 
		  MemIntType* l_cAddr, 
		  unsigned int l_aColBlocks, 
		  unsigned int l_aRowBlocks,
		  unsigned int l_bColBlocks, 
		  unsigned int l_aLd, 
		  unsigned int l_bLd,
		  unsigned int l_cLd
		  );
    void uut_top_test( MemIntType* l_aAddr, 
		  MemIntType* l_bAddr, 
		  MemIntType* l_cAddr, 
		  unsigned int l_aColBlocks, 
		  unsigned int l_aRowBlocks,
		  unsigned int l_bColBlocks, 
		  unsigned int l_aLd, 
		  unsigned int l_bLd,
		  unsigned int l_cLd, 
		  unsigned int l_transpBlocks
		  );
}

#endif // UUT_TOP_HPP
