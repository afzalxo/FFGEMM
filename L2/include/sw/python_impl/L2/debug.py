from matrix import Matrix


def do_debug(): 
    t_ColMemWords = 4
    t_RowMemWords = 4
    t_MemWidth = 2
    t_MH = t_MemWidth * t_RowMemWords
    BLAS_m = 32
    BLAS_k = 32
    BLAS_n = 32
    matA = Matrix(BLAS_m, BLAS_k)
    matA.fillModDebug(13, 1)
    matB = Matrix(BLAS_k, BLAS_n)
    matB.fillModDebug(13, 1)
    matC = Matrix(BLAS_m, BLAS_n)
    matC.zero()

    # print(matA)
    # print(matB)
    matC = matA * matB
    print(matC.block(0, 0, t_MH, t_MH))

    # subA = matA.block(0, 0, block_size_rows = t_MH, block_size_cols = t_MemWidth * t_ColMemWords) + matA.block(1, 1, block_size_rows = t_MH, block_size_cols = t_MemWidth * t_ColMemWords) + matA.block(2, 2, block_size_rows = t_MH, block_size_cols = t_MemWidth * t_ColMemWords) + matA.block(3, 3, block_size_rows = t_MH, block_size_cols = t_MemWidth * t_ColMemWords)
    # subB = matB.block(0, 0, block_size_rows = t_MH, block_size_cols = t_MemWidth * t_ColMemWords) + matB.block(1, 1, block_size_rows = t_MH, block_size_cols = t_MemWidth * t_ColMemWords) + matB.block(2, 2, block_size_rows = t_MH, block_size_cols = t_MemWidth * t_ColMemWords) + matB.block(3, 3, block_size_rows = t_MH, block_size_cols = t_MemWidth * t_ColMemWords)
    # subC = subA * subB
    # print(subC)

if __name__ == "__main__":
    do_debug()
