import random


class Matrix:
    def __init__(self, t_Rows, t_Cols, dtype=int, randomize=False):
        self.t_Rows = t_Rows
        self.t_Cols = t_Cols
        self.dtype = dtype
        self.matrix = [[dtype() for _ in range(t_Cols)] for _ in range(t_Rows)]
        if randomize:
            self.randomize()

    # Overload [] operator for accessing matrix elements
    def __getitem__(self, p_Row: int, p_Col=None):
        if p_Col is None:
            return self.matrix[p_Row]
        else:
            return self.matrix[p_Row][p_Col]

    def __getrowmajor__(self, p_Index: int):
        assert p_Index < self.t_Rows * self.t_Cols and p_Index >= 0
        return self.matrix[p_Index // self.t_Cols][p_Index % self.t_Cols]

    # Overload item assignment for matrix rows
    def __setitem__(self, p_Row: int, p_Col=None, p_Val=None):
        if p_Col is None and isinstance(p_Val, list):
            self.matrix[p_Row] = p_Val
        elif p_Col is not None and p_Val is not None:
            self.matrix[p_Row][p_Col] = p_Val
        else:
            raise TypeError("Invalid assignment to matrix")

    # Overload print operator for printing matrix
    # Set width of each element to 4
    def __repr__(self):
        # find max width of elements
        l_max = self.max()
        l_min = self.min()
        l_max = max(abs(l_max), abs(l_min))
        l_width = len(str(l_max)) + 2

        l_str = ""
        for i in range(self.t_Rows):
            for j in range(self.t_Cols):
                l_str += "{:{width}}".format(self.matrix[i][j], width=l_width)
            l_str += "\n"
        return l_str

    def randomize(self, low=None, high=None):
        if low is None:
            low = -10
        if high is None:
            high = 10
        for i in range(self.t_Rows):
            for j in range(self.t_Cols):
                self.matrix[i][j] = random.randint(low, high)
        return self

    def zero(self):
        for i in range(self.t_Rows):
            for j in range(self.t_Cols):
                self.matrix[i][j] = self.dtype()
        return self

    def clear(self):
        return self.zero()

    def __eq__(self, other):
        return self.matrix == other.matrix

    def __add__(self, other):
        l_sum = Matrix(self.t_Rows, self.t_Cols)
        for i in range(self.t_Rows):
            for j in range(self.t_Cols):
                l_sum[i][j] = self.matrix[i][j] + other.matrix[i][j]
        return l_sum

    def __sub__(self, other):
        l_diff = Matrix(self.t_Rows, self.t_Cols)
        for i in range(self.t_Rows):
            for j in range(self.t_Cols):
                l_diff[i][j] = self.matrix[i][j] - other.matrix[i][j]
        return l_diff

    def __mul__(self, other):
        l_prod = Matrix(self.t_Rows, self.t_Cols)
        assert isinstance(other, Matrix)
        assert self.t_Cols == other.t_Rows
        for i in range(self.t_Rows):
            for j in range(other.t_Cols):
                for k in range(self.t_Cols):
                    l_prod[i][j] += self.matrix[i][k] * other.matrix[k][j]
        return l_prod

    def submatrix(self, row_start, num_rows, col_start, num_cols):
        assert row_start + num_rows <= self.t_Rows
        assert col_start + num_cols <= self.t_Cols
        l_sub = Matrix(num_rows, num_cols)
        for i in range(num_rows):
            for j in range(num_cols):
                l_sub[i][j] = self.matrix[row_start + i][col_start + j]
        return l_sub

    # Get block of size block_size_rows x block_size_cols
    # Where block is defined as a submatrix of the matrix
    # block_row and block_col are the row and column of the block to get
    # Blocks are indexed row-wise
    # If block_size_rows and block_size_cols are not specified, then the submatrixes
    # are of size t_Rows // 2 x t_Cols // 2
    def block(self, block_row, block_col, block_size_rows=None, block_size_cols=None):
        if block_size_rows is None:
            block_size_rows = self.t_Rows // 2
        if block_size_cols is None:
            block_size_cols = self.t_Cols // 2
        assert block_row < self.t_Rows // block_size_rows
        assert block_col < self.t_Cols // block_size_cols
        assert self.t_Rows % block_size_rows == 0
        assert self.t_Cols % block_size_cols == 0
        l_block = Matrix(block_size_rows, block_size_cols)
        for i in range(block_size_rows):
            for j in range(block_size_cols):
                l_block[i][j] = self.matrix[block_row * block_size_rows + i][
                    block_col * block_size_cols + j
                ]
        return l_block

    def fillMod(self, p_Max, p_First=0):
        l_val = p_First
        for row in range(self.t_Rows):
            for col in range(self.t_Cols):
                self.matrix[row][col] = l_val
                l_val += 1
                l_val %= p_Max

    def fillModDebug(self, p_Max, p_First=0):
        for row in range(self.t_Rows):
            for col in range(self.t_Cols):
                self.matrix[row][col] = (row * self.t_Cols + col) % p_Max + p_First

    # Max element in matrix
    def max(self):
        l_max = self.matrix[0][0]
        for i in range(self.t_Rows):
            for j in range(self.t_Cols):
                if self.matrix[i][j] > l_max:
                    l_max = self.matrix[i][j]
        return l_max

    # Min element in matrix
    def min(self):
        l_min = self.matrix[0][0]
        for i in range(self.t_Rows):
            for j in range(self.t_Cols):
                if self.matrix[i][j] < l_min:
                    l_min = self.matrix[i][j]
        return l_min

    @property
    def __len__(self):
        return self.t_Rows

    @property
    def __rows__(self):
        return self.t_Rows

    @property
    def __cols__(self):
        return self.t_Cols

    # Overload shape operator for getting shape of matrix
    @property
    def shape(self):
        return (self.t_Rows, self.t_Cols)

    def transpose(self):
        # Transpose self.matrix
        self.matrix = [list(i) for i in zip(*self.matrix)]
        self.t_Rows, self.t_Cols = self.t_Cols, self.t_Rows
        return self

    def map(self, func: callable):
        for i in range(self.t_Rows):
            for j in range(self.t_Cols):
                self.matrix[i][j] = func(self.matrix[i][j])

    def initialize_incremental(self):
        for i in range(self.t_Rows):
            for j in range(self.t_Cols):
                self.matrix[i][j] = i * self.t_Cols + j

    def place_tile(self, tile: list, index: int):
        tile_size = len(tile)
        # calculate the row and column of the start of tile
        # Tiles are part of blocks, each block of dimension tile_size x tile_size
        # Blocks across cols go from left to right, indexed 0 to t_Cols // tile_size
        # Blocks across rows go from top to bottom, indexed 0 to t_Rows // tile_size
        # Blocks are indexed row-wise
        # The block index of the tile is index // tile_size
        block_index = index // tile_size
        block_row = block_index // (self.t_Cols // tile_size)
        block_col = block_index % (self.t_Cols // tile_size)
        # Find the row and column of the start of the tile relative to the matrix
        row = block_row * tile_size + index % tile_size
        col = block_col * tile_size
        # Place the tile
        for i in range(tile_size):
            self.matrix[row][col + i] = tile[i]

    def as_list(self) -> list:
        # Return the matrix as a flattened list
        return [
            self.matrix[i][j] for i in range(self.t_Rows) for j in range(self.t_Cols)
        ]

    '''
    def to_stream(self, t_MemWidth: int) -> Stream:
        outstream = Stream(dtype=WideType)
        matrix_as_list = self.as_list()
        for elem in matrix_as_list:
            memword = WideType(t_MemWidth, dtype=self.dtype)
            for i in range(t_MemWidth):
                memword[i] = elem
            outstream.write(memword)
        return outstream
    '''
