from matrix import Matrix
from xf_types import WideType


class GEMMMatAMover:
    def __init__(self, matrixA: Matrix, p_n: int, t_M: int):
        self.matrixA = matrixA
        self.t_M = t_M

        self.m = 0
        self.r = 0
        self.k = 0

        self.p_k = matrixA.shape[0]
        self.p_m = matrixA.shape[1]
        self.p_n = p_n

        self.l_iter = self.p_m // t_M
        self.l_repeat = self.p_n // t_M

        assert self.p_m % t_M == 0, "p_m must be divisible by t_M"
        assert self.p_n % t_M == 0, "p_n must be divisible by t_M"

    # Overload __next__ to allow for obtaining dataset
    def __next__(self) -> WideType:
        l_A = WideType(self.t_M)
        for i in range(self.t_M):
            l_A[i] = self.matrixA.__getrowmajor__(
                (self.k * self.l_iter + self.m) * self.t_M + i)

        self.k += 1
        if self.k == self.p_k:
            self.k = 0
            self.r += 1
            if self.r == self.l_repeat:
                self.r = 0
                self.m += 1
                if self.m == self.l_iter:
                    self.m = 0

        return l_A


class GEMMMatBMover:
    def __init__(self, matrixB: Matrix, p_m: int, t_M: int):
        self.matrixB = matrixB
        self.t_M = t_M

        self.n = 0
        self.r = 0
        self.k = 0

        self.p_k = matrixB.shape[0]
        self.p_n = matrixB.shape[1]
        self.p_m = p_m

        self.l_repeat = self.p_m // t_M
        self.l_iter = self.p_n // t_M

        assert self.p_m % t_M == 0, "p_m must be divisible by t_M"
        assert self.p_n % t_M == 0, "p_n must be divisible by t_M"

    # Overload __next__ to allow for obtaining dataset
    def __next__(self) -> WideType:
        l_B = WideType(self.t_M)
        for i in range(self.t_M):
            l_B[i] = self.matrixB.__getrowmajor__(
                (self.k * self.l_iter + self.n) * self.t_M + i)

        self.k += 1
        if self.k == self.p_k:
            self.k = 0
            self.n += 1
            if self.n == self.l_iter:
                self.n = 0
                self.r += 1
                if self.r == self.l_repeat:
                    self.r = 0

        return l_B


class GEMMMatStreamToMatrix:
    def __init__(self, p_m: int, p_n: int, t_M: int):
        self.t_M = t_M
        self.p_m = p_m
        self.p_n = p_n
        self.matrixC = Matrix(p_m, p_n)
        self.i = 0

    def StreamToMatrix(self, l_C: list):
        self.matrixC.place_tile(l_C, self.i)
        self.i += 1

    def GetMatrix(self) -> Matrix:
        return self.matrixC
