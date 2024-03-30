import copy
from matrix import Matrix


class WideType:
    def __init__(self, t_Width=1, other=None, dtype=None):
        if other is None and dtype is None:
            self.m_Val = [0 for _ in range(t_Width)]
        elif other is None and dtype is not None:
            self.m_Val = [dtype() for _ in range(t_Width)]
        elif isinstance(other, WideType):
            self.m_Val = other.m_Val
            self.dtype = other.dtype
        else:
            raise TypeError("Invalid type for WideType constructor")

    def from_list(self, p_List: list):
        self.m_Val = p_List

    def getVal(self, i):
        return self.m_Val[i]

    def setVal(self, i, val):
        self.m_Val[i] = val

    # Overload [] operator for easy access
    def __getitem__(self, i):
        return self.m_Val[i]

    # Overload item assignment
    def __setitem__(self, i, val):
        self.m_Val[i] = val

    # equality operator
    def __eq__(self, other):
        return self.m_Val == other.m_Val

    # shift
    def shift(self, p_ValIn=0):
        # Shift all values to the right
        # and set the first value to p_ValIn
        l_valOut = self.m_Val[len(self.m_Val) - 1]
        for i in range(len(self.m_Val) - 1, 0, -1):
            self.m_Val[i] = self.m_Val[i - 1]
        self.m_Val[0] = p_ValIn
        return l_valOut

    def zero(self):
        for i in range(len(self.m_Val)):
            self.m_Val[i] = 0
        return self

    def __repr__(self):
        return str(self.m_Val)

    def __str__(self):
        return str(self.m_Val)

    # as list
    def as_list(self):
        return self.m_Val


class TaggedFloat:
    def __init__(self, t_Float=0, t_Tag=0):
        self.m_Tag = t_Tag
        self.m_Float = t_Float

    def getTag(self):
        return self.m_Tag

    def getFloat(self):
        return self.m_Float

    def setTag(self, t_Tag):
        self.m_Tag = t_Tag

    def setFloat(self, t_Float):
        self.m_Float = t_Float

    def __repr__(self):
        return str(self.m_Float)
        # return str(self.m_Tag) + " " + str(self.m_Float)

    def __str__(self):
        return str(self.m_Float)
        # return str(self.m_Tag) + " " + str(self.m_Float)

    def __getitem__(self):
        return self.m_Float


class TriangSrl:
    def __init__(self, t_Width, dtype=int):
        self.t_Width = t_Width
        self.ap_shift_reg = [[dtype() for _ in range(t_Width)] for _ in range(t_Width)]
        self.dtype = dtype

    def shift(self, p_DiagIn: WideType) -> WideType:
        l_edgeOut = WideType(self.t_Width, dtype=self.dtype)
        for i in range(self.t_Width):
            l_edgeOut.setVal(i, self.ap_shift_reg[i][i])
            # Shift all values of self.ap_shift_reg[i] to the right
            # and set the first value to p_DiagIn[i]
            for j in range(self.t_Width - 1, 0, -1):
                self.ap_shift_reg[i][j] = self.ap_shift_reg[i][j - 1]
            self.ap_shift_reg[i][0] = p_DiagIn[i]
        return l_edgeOut

    def clear(self):
        for i in range(self.t_Width):
            for j in range(self.t_Width):
                self.ap_shift_reg[i][j] = 0

    def __repr__(self):
        # return string containing lower triangle of self.ap_shift_reg
        # and - for upper triangle
        l_str = ""
        for i in range(self.t_Width):
            for j in range(self.t_Width):
                if i >= j:
                    l_str += str(self.ap_shift_reg[i][j]) + " "
                else:
                    l_str += "- "
            l_str += "\n"
        return l_str


class WindowRm:
    def __init__(self, t_Rows, t_Cols, dtype=int):
        self.t_Rows = t_Rows
        self.t_Cols = t_Cols
        self.m_Val = [WideType(t_Cols, dtype=dtype) for _ in range(t_Rows)]

    def getVal(self, p_Row, p_Col):
        return self.m_Val[p_Row][p_Col]

    # Overload [] operator for easy access
    def __getitem__(self, p_Row) -> WideType:
        return self.m_Val[p_Row]

    def clear(self):
        for i in range(self.t_Rows):
            for j in range(self.t_Cols):
                self.m_Val[i].setVal(j, 0)

    def __repr__(self):
        l_str = ""
        for i in range(self.t_Rows):
            for j in range(self.t_Cols):
                l_str += str(self.m_Val[i][j]) + " "
            l_str += "\n"
        return l_str

    # Down shift (0th row in, the last row out)
    def shift(self, p_EdgeIn: WideType) -> WideType:
        l_valOut = self.m_Val[self.t_Rows - 1]
        for i in range(self.t_Rows - 1, 0, -1):
            self.m_Val[i] = self.m_Val[i - 1]
        self.m_Val[0] = copy.deepcopy(p_EdgeIn)
        return l_valOut

    # Right shift (0th col in, the last col out)
    def shift_right(self, p_EdgeIn: WideType) -> WideType:
        p_EdgeIn = copy.deepcopy(p_EdgeIn)
        l_valOut = WideType(self.t_Rows)
        for i in range(self.t_Rows):
            l_valOut.setVal(i, self.m_Val[i].getVal(self.t_Cols - 1))
            self.m_Val[i].shift(p_EdgeIn.getVal(i))
        return l_valOut


class MemIntType:
    def __init__(self, t_MemWidth):
        self.t_MemWidth = t_MemWidth
        self.m_Val = []

    def from_matrix(self, p_Matrix: Matrix):
        dtype = p_Matrix.dtype
        matrix_as_list = p_Matrix.as_list()
        # Iterate over the list and put each t_MemWidth elements
        # into a WideType and append it to self.m_Val
        for i in range(0, len(matrix_as_list), self.t_MemWidth):
            memword = WideType(self.t_MemWidth, dtype=dtype)
            memword.from_list(matrix_as_list[i: i + self.t_MemWidth])
            self.m_Val.append(memword)

    def to_matrix(self, nRows: int, nCols: int) -> Matrix:
        assert nRows * nCols == len(self.m_Val) * self.t_MemWidth, "Matrix dimensions do not match"
        matrix = Matrix(nRows, nCols)   # TODO: dtype needs to be passed
        i, j = 0, 0
        for memword in self.m_Val:
            for k in range(self.t_MemWidth):
                matrix[i][j] = memword[k]
                j += 1
                if j == nCols:
                    j = 0
                    i += 1
        return matrix

    def __getitem__(self, i):
        return self.m_Val[i]

    def __setitem__(self, i, val: WideType):
        self.m_Val[i] = val

    def __len__(self):
        return len(self.m_Val)

    def __repr__(self):
        return str(self.m_Val)

    def __str__(self):
        return str(self.m_Val)

    def allocate(self, p_NumWords):
        self.m_Val = [WideType(self.t_MemWidth) for _ in range(p_NumWords)]
