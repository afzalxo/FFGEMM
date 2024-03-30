from xf_types import WideType, WindowRm, TriangSrl
from matrix import Matrix
from gemmMatMover import GEMMMatAMover, GEMMMatBMover

from gemm import Gemm


def main():
    print("###"*10)
    print_matrices = False
    BLAS_m = 16
    BLAS_n = 16
    BLAS_k = 16
    BLAS_parEntries = 4
    l_blocks = (BLAS_m * BLAS_n) // (BLAS_parEntries * BLAS_parEntries)
    matA = Matrix(BLAS_m, BLAS_k)
    matA.initialize_incremental()
    matB = Matrix(BLAS_k, BLAS_n)
    matB.initialize_incremental()
    true_matC = matA * matB
    if print_matrices:
        print('Matrix A:')
        print(matA)
        print('Matrix B:')
        print(matB)
        print('True Matrix C:')
        print(true_matC)
    moverA = GEMMMatAMover(matA.transpose(), BLAS_n, t_M=BLAS_parEntries)
    moverB = GEMMMatBMover(matB, BLAS_m, t_M=BLAS_parEntries)
    gemm = Gemm(moverA, moverB, p_blocks=l_blocks, t_KBufferDim=BLAS_k, t_ParEntries=BLAS_parEntries)
    kernel_matC = gemm.compute()
    if print_matrices:
        print('Kernel Matrix C:')
        print(kernel_matC)

    assert (kernel_matC == true_matC), 'Matrices are not equal!. Failed test.'
    print('Test passed!')


if __name__ == '__main__':
    main()
