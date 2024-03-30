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

#ifndef XF_BLAS_KERNEL_HPP
#define XF_BLAS_KERNEL_HPP

/*
 * @file kernel.hpp
 */

#include "mem.hpp"
#include "memType.hpp"
#include "gemmKernel.hpp"
// #include "strassensKernel.hpp"
#include "strassensSquaredKernel.hpp"

// Compute engine types
#if BLAS_runGemm == 1
/*
typedef xf::blas::GemmKernel<BLAS_dataType,
                             BLAS_XdataType,
                             BLAS_memWidth,
                             BLAS_XmemWidth,
                             BLAS_gemmKBlocks,
                             BLAS_gemmMBlocks,
                             BLAS_gemmNBlocks>
    GemmType;
typedef xf::blas::StrassensKernel<BLAS_dataType,
                             	  BLAS_memWidth,
                             	  BLAS_gemmKBlocks,
                             	  BLAS_gemmMBlocks,
                             	  BLAS_gemmNBlocks>
    GemmType;
*/
typedef xf::blas::StrassensSquaredKernel<BLAS_dataType,
                             	  BLAS_memWidth,
                             	  BLAS_gemmKBlocks,
                             	  BLAS_gemmMBlocks,
                             	  BLAS_gemmNBlocks>
    GemmType;

#endif

#if BLAS_runTranspose == 1
typedef xf::blas::TransposeKernel<BLAS_dataType,
                                  BLAS_memWidth,
                                  BLAS_gemmMBlocks,
                                  BLAS_gemmKBlocks>
    TransposeType;
#endif

typedef xf::blas::TimeStamp<BLAS_numInstr> TimeStampType;

/**
 * @brief blasKernel is the uniform top function for blas function kernels with interfaces to DDR/HBM memories
 *
 * @param p_MemRd the memory port for data loading
 * @param p_MemWr the memory port for data writing
 *
 */
extern "C" {
   void blasKernel(MemIntType* p_MemRd, MemIntType* p_MemWr);
} // extern C

#endif
