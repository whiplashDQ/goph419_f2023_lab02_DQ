import numpy as np
from linalg_interp import gauss_iter_solve

def main():
    
    # test gauss_iter_solve function 
    A = np.array([[7, 1, 2],
                  [2, 7, 1],
                  [-1, -2, 3]], dtype=float)

    b = np.array([7, -6, -17], dtype=float)

    x_seidel = gauss_iter_solve(A, b, tol = 1e-8, alg = 'seidel')
    print("Solution using Gauss-Seidel:", x_seidel)
    x_jacobi = gauss_iter_solve(A, b, tol = 1e-8, alg = 'jacobi')
    print("Solution using Jacobi:", x_jacobi)
    solution1 = np.linalg.solve(A, b)
    print("Using linalg.solve to solve the system equations: ",solution1)

    # another test
    A = np.array([[5, 6, 7],
                  [2, -7, 1],
                  [3, -1, 5]], dtype=float)

    b = np.array([8, 16, -17], dtype=float)

    x_seidel = gauss_iter_solve(A, b, tol = 1e-8, alg = 'seidel')
    print("Solution using Gauss-Seidel:", x_seidel)
    x_jacobi = gauss_iter_solve(A, b, tol = 1e-8, alg = 'jacobi')
    print("Solution using Jacobi:", x_jacobi)
    solution2 = np.linalg.solve(A, b)
    print("Using linalg.solve to solve the system equations: ",solution2)

    A = np.array([[7, 1, 2],
                  [2, 7, 1],
                  [-1, -2, 3]], dtype=float)

    tol = 1e-8
    test_inverse(A, tol, 'seidel')
    test_inverse(A, tol, 'jacobi')
   
# test x is the inverse of A
def test_inverse(A, tol, alg):
    n = A.shape[0]
    identity_matrix = np.eye(n)
    A_inv = np.zeros_like(A)

    # Solve for each column of A^-1
    for i in range(n):
        A_inv[:, i] = gauss_iter_solve(A, identity_matrix[:, i], tol, alg)

    # Check if AA^-1 is close to the identity matrix
    AA_inv = np.dot(A, A_inv)    
    print('check AA-1 is close to Identity matrix',AA_inv)
if __name__ == "__main__":
    main()
    


