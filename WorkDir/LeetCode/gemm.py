import numpy as np

def matmul(A, B):
    """Matrix multiplication of A and B."""
    return np.dot(A, B)

if __name__ == "__main__":
    # Define two matrices
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8], [9, 10], [11, 12]])

    print("Matrix A:")
    print(A)
    print("Matrix B:")
    print(B)

    result = matmul(A, B)

    print("Result of A * B:")
    print(result)
