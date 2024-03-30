import numpy as np

from xf_types import WideType, MemIntType, WindowRm, TriangSrl
from matrix import Matrix
from stream import Stream

from matrixBuffer import MatrixBuffer
from transpose import Transpose
from gemm import Gemm
from gemmKernel import GemmKernel
from strassensKernel import StrassensKernel

from submatrixOps import SubMatrixOps


def main():
    print("###"*10)
    BLAS_dataType = int
    print_matrices = False
    BLAS_m = 128
    BLAS_n = 128
    BLAS_k = 128
    BLAS_parEntries = 4
    BLAS_gemmMBlocks = 4
    BLAS_gemmKBlocks = 4
    BLAS_gemmNBlocks = 4
    BLAS_memWidth = 16
    matA = Matrix(BLAS_m, BLAS_k)
    matA.fillMod(67, 1)
    matB = Matrix(BLAS_k, BLAS_n)
    matB.fillMod(129, 65)
    matC = Matrix(BLAS_m, BLAS_n)
    matC.zero()
    true_matC = matA * matB
    if print_matrices:
        print('Matrix A:')
        print(matA)
        print('Matrix B:')
        print(matB)
        print('True Matrix C:')
        print(true_matC)

    l_aColBlocks = BLAS_k // (BLAS_memWidth * BLAS_parEntries)
    l_aRowBlocks = BLAS_m // (BLAS_memWidth * BLAS_parEntries)
    l_bColBlocks = BLAS_n // (BLAS_memWidth * BLAS_parEntries)

    l_transpBlocks = l_aColBlocks * l_aRowBlocks * l_bColBlocks * BLAS_parEntries

    '''
    streamA = Stream(dtype=WideType)
    streamA = streamA.from_matrix(BLAS_dataType, BLAS_memWidth, matA)
    # streamB = Stream(dtype=WideType)
    # streamB = streamB.from_matrix(BLAS_dataType, BLAS_memWidth, matB)

    # gemm_inst = Gemm(BLAS_dataType, BLAS_memWidth, BLAS_parEntries * BLAS_memWidth, BLAS_parEntries, BLAS_parEntries)

    # streamC = gemm_inst.gemm(streamA, streamB, BLAS_parEntries ** 2)
    # print(streamC)
    streamOut = Stream(dtype=WideType)

    transposeOp = Transpose(BLAS_dataType, BLAS_parEntries, BLAS_parEntries, BLAS_parEntries, l_transpBlocks, BLAS_n // BLAS_memWidth)
    streamOut = transposeOp.process(streamA, streamOut)
    print(streamOut)

    l_abBlocks = l_aColBlocks * l_aRowBlocks * l_bColBlocks
    streamA = Stream(dtype=WideType)
    streamA = streamA.from_matrix(BLAS_dataType, BLAS_memWidth, matA)
    streamOut = Stream(dtype=WideType)
    matrixBufferOp = MatrixBuffer(BLAS_dataType, BLAS_memWidth, BLAS_memWidth * BLAS_parEntries, BLAS_parEntries)
    streamOut = matrixBufferOp.process(streamA, streamOut, l_abBlocks, BLAS_parEntries)
    print(streamOut)
    '''

    aAddr = MemIntType(BLAS_memWidth)
    aAddr.from_matrix(matA)
    bAddr = MemIntType(BLAS_memWidth)
    bAddr.from_matrix(matB)
    cAddr = MemIntType(BLAS_memWidth)
    cAddr.from_matrix(matC)

    # gemmKernelOp = GemmKernel(BLAS_dataType, BLAS_memWidth, BLAS_gemmKBlocks, BLAS_gemmMBlocks, BLAS_gemmNBlocks)
    gemmKernelOp = StrassensKernel(BLAS_dataType, BLAS_memWidth, BLAS_gemmKBlocks, BLAS_gemmMBlocks, BLAS_gemmNBlocks)

    res = gemmKernelOp.runGemm(aAddr, bAddr, cAddr, BLAS_m, BLAS_n, BLAS_k, BLAS_k, BLAS_n, BLAS_n)
    resMat = res.to_matrix(BLAS_m, BLAS_n)
    if print_matrices:
        print('Kernel Output Matrix:')
        print(resMat)

    if true_matC == resMat:
        print('Kernel and True Matrix Match')
        print('Test Passed!')
    else:
        print('Kernel and True Matrix Do Not Match')
        print('Test Failed!')


if __name__ == '__main__':
    main()
