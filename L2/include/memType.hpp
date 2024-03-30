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

#ifndef XF_BLAS_DDRTYPE_HPP
#define XF_BLAS_DDRTYPE_HPP

#include "mem.hpp"
#include "kargs.hpp"

// Location of code aand data segements in external memory
#define BLAS_codePage 0
#define BLAS_resPage 1
#define BLAS_dataPage 2

// Page and instruction sizes
#define BLAS_pageSizeBytes 4096

// Memory interface types
typedef xf::blas::MemUtil<BLAS_dataType, BLAS_memWidth, BLAS_memWidth * sizeof(BLAS_dataType) * 8> MemUtilType;
typedef MemUtilType::MemWideType MemType;
typedef typename MemType::t_TypeInt MemIntType;

// VLIV processing types
typedef xf::blas::Kargs<BLAS_dataType, BLAS_memWidth, BLAS_argInstrWidth, BLAS_argPipeline> KargsType;

typedef KargsType::MemInstrType KargsMemInstrType; // 512 bit wide type across all DDR-width architectures

#endif
