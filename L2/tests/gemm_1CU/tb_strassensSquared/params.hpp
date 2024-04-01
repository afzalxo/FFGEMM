#ifndef XF_BLAS_PARAMS_HPP
#define XF_BLAS_PARAMS_HPP

#include "types.hpp"

using namespace xf::blas;

#define BLAS_dataType float

constexpr unsigned int BLAS_m = 256;
constexpr unsigned int BLAS_k = 256;
constexpr unsigned int BLAS_n = 256;

constexpr unsigned int BLAS_lda = BLAS_k;
constexpr unsigned int BLAS_ldb = BLAS_n;
constexpr unsigned int BLAS_ldc = BLAS_n;

constexpr unsigned int BLAS_memWidth = 16;
constexpr unsigned int BLAS_gemmMBlocks = 4;
constexpr unsigned int BLAS_gemmKBlocks = 4;
constexpr unsigned int BLAS_gemmNBlocks = 4;

typedef WideType<BLAS_dataType, BLAS_memWidth> MemWideType;
typedef typename MemWideType::t_TypeInt MemIntType;

#define RUN_STRASSENS 1
// #define RUN_BASELINE 1

#endif  // PARAMS_HPP
