import sys
import struct


class BinReader:
    def __init__(self, filename: str, m: int, k: int, n: int, dataType):
        self.filename = filename
        self.f = open(filename, 'rb')
        self.pageSizeBytes = 4096
        self.dataType = dataType
        self.offset = 4096 * 3  # skip the first 3 pages
        self.m = m
        self.k = k
        self.n = n

    # MatrixA size: m x k elements, each of size sizeof(dataType)
    # MatrixB size: k x n elements, each of size sizeof(dataType)
    # MatrixC size: m x n elements, each of size sizeof(dataType)
    def readMatrixA(self):
        matrixA = []
        for i in range(self.m):
            row = []
            for j in range(self.k):
                elem = self.readElement()
                row.append(elem)
                if i == 0 and j < 10:
                    print(elem)
            matrixA.append(row)
        return matrixA

    def readMatrixB(self):
        matrixB = []
        for i in range(self.k):
            row = []
            for j in range(self.n):
                row.append(self.readElement())
            matrixB.append(row)
        return matrixB

    def readElement(self):
        self.f.seek(self.offset)
        self.offset += 4  # sys.getsizeof(self.dataType())
        dat = self.f.read(4)
        return struct.unpack('f', dat)[0]

    def close(self):
        self.f.close()

    def readMatrixC(self):
        matrixC = []
        for i in range(self.m):
            row = []
            for j in range(self.n):
                row.append(self.readElement())
            matrixC.append(row)
        return matrixC


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Usage: python3 readBin.py <filename> <m> <k> <n>, <dataType>')
        exit(1)
    filename = sys.argv[1]
    m = int(sys.argv[2])
    k = int(sys.argv[3])
    n = int(sys.argv[4])
    dataType = sys.argv[5]
    if dataType == 'float':
        dataType = float
    elif dataType == 'int':
        dataType = int
    else:
        print('Error: dataType must be either float or int')
        exit(1)

    reader = BinReader(filename, m, k, n, dataType)
    matA = reader.readMatrixA()
