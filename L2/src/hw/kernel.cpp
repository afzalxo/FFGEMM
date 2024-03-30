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

#include <assert.h>
#include "kernel.hpp"

// The extern C still needed - otherwise cpu emu fails
//   prj_sda.exe: symbol lookup error: ./dltmp: undefined symbol: kernelSgemm
extern "C" {

void kernelOp(MemIntType* p_MemRd,
              MemIntType* p_MemWr,
              //    hls::stream<TimeStampType::OpType> &p_Control,
              hls::stream<TimeStampType::TimeType>& p_Time) {
#pragma HLS INLINE self off

    typedef KargsType::OpType KargsOpType;
    typedef xf::blas::ControlArgs ControlArgsType;
#if BLAS_runGemm == 1
    GemmType l_gemm;
    typedef GemmType::GemmArgsType GemmArgsType;
#endif

    ///////////////////////////////////////////////////////////////////////////
    // VLIW op decoding
    ///////////////////////////////////////////////////////////////////////////
    unsigned int l_pc = 0;
    bool l_isLastOp = false;
    static const unsigned int l_tsDepth = TimeStampType::t_FifoDepth;

    // Checks for code, result, and data segment sizes
    KargsMemInstrType l_code[BLAS_numInstr], l_res[BLAS_numInstr];
//#pragma HLS ARRAY_PARTITION variable=l_code complete dim=0
//#pragma HLS ARRAY_PARTITION variable=l_res  complete dim=0
#ifndef __SYNTHESIS__
    assert(sizeof(l_code) <= (BLAS_resPage - BLAS_codePage) * BLAS_pageSizeBytes);
    assert(sizeof(l_code) <= (BLAS_dataPage - BLAS_resPage) * BLAS_pageSizeBytes);
#endif

    // Prefetch all instructions for more accurate cycle measurements
    for (unsigned int l_pc = 0; l_pc < BLAS_numInstr; ++l_pc) {
        l_code[l_pc].loadFromMem(p_MemRd, BLAS_codePage * MemType::per4k() + l_pc * KargsType::getInstrWidth());
    }

    // Decode and execute
    TimeStampType::TimeType l_tsPrev = 0;
    KargsType l_kargsRes;
    for (unsigned int l_pc = 0; l_pc < BLAS_numInstr; ++l_pc) {
        KargsType l_kargs;
        KargsOpType l_op = l_kargs.loadFromInstr(l_code[l_pc]);
        switch (l_op) {
            case KargsType::OpControl: {
                ControlArgsType l_controlArgs = l_kargs.getControlArgs();
                l_isLastOp = l_controlArgs.getIsLastOp();
#ifndef __SYNTHESIS__
                assert(!l_isLastOp || (l_pc == BLAS_numInstr - 1));
#endif
                break;
            }
#if BLAS_runGemm == 1
            case KargsType::OpGemm: {
                GemmArgsType l_gemmArgs = l_kargs.getGemmArgs();
                if (BLAS_runGemm) l_gemm.runGemm(p_MemRd, p_MemWr, l_gemmArgs);
                // if (BLAS_runGemm) l_gemm.runMatrixAdd(p_MemRd, p_MemWr, l_gemmArgs);
                break;
            }
#endif
            default: {
#ifndef __SYNTHESIS__
                assert(false);
#endif
            }
        }

        // Collect and store cycle count
        TimeStampType::TimeType l_ts = p_Time.read();
        if (l_pc >= l_tsDepth) {
            xf::blas::InstrResArgs l_instrRes(l_tsPrev, reg(l_ts));
            l_kargsRes.setInstrResArgs(l_instrRes);
            l_kargsRes.storeToInstr(l_res[l_pc - l_tsDepth]);
        }
        l_tsPrev = reg(l_ts);
    }

    for (unsigned int l_d = 0; l_d < l_tsDepth; ++l_d) {
        TimeStampType::TimeType l_ts = p_Time.read();
        xf::blas::InstrResArgs l_instrRes(l_tsPrev, l_ts);
        l_kargsRes.setInstrResArgs(l_instrRes);
        l_kargsRes.storeToInstr(l_res[BLAS_numInstr - l_tsDepth + l_d]);
        l_tsPrev = l_ts;
    }

    // Store instruction results in DDR result segment
    for (unsigned int l_pc = 0; l_pc < BLAS_numInstr; ++l_pc) {
        l_res[l_pc].storeToMem(p_MemWr, BLAS_resPage * MemType::per4k() + l_pc * KargsType::getInstrWidth());
    }
}

/**
 * @brief blasKernel defines the kernel top function, with DDR/HBM as an interface
 *
 * @param p_MemRd is DDR/HBM memory address used for read
 * @param p_MemWr is DDR/HBM memory address used for write
 */
void blasKernel(MemIntType* p_MemRd, MemIntType* p_MemWr) {
#pragma HLS INTERFACE m_axi port = p_MemRd offset = slave bundle = gmemm num_write_outstanding = \
    16 num_read_outstanding = 16 max_write_burst_length = 16 max_read_burst_length = 16 depth = 16 latency = 125
#pragma HLS INTERFACE m_axi port = p_MemWr offset = slave bundle = gmemm num_write_outstanding = \
    16 num_read_outstanding = 16 max_write_burst_length = 16 max_read_burst_length = 16 depth = 16 latency = 125

#pragma HLS INTERFACE s_axilite port = p_MemRd bundle = control
#pragma HLS INTERFACE s_axilite port = p_MemWr bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATA_PACK variable = p_MemRd
#pragma HLS DATA_PACK variable = p_MemWr

#if TEST_MEMCPY
    p_MemWr[0] = p_MemRd[0];
#else
    TimeStampType l_tr;
    // hls::stream<TimeStampType::OpType> l_controlStream;
    hls::stream<TimeStampType::TimeType> l_timeStream;
//#pragma HLS STREAM   variable=l_controlStream  depth=1
#pragma HLS STREAM variable = l_timeStream depth = 1

#pragma HLS DATAFLOW

    l_tr.runTs(/*l_controlStream, */ l_timeStream);
    kernelOp(p_MemRd, p_MemWr, /*l_controlStream,*/ l_timeStream);
#endif
}

} // extern C
