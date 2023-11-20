import numpy as np



def gauss_iter_solve(A,b,tol,alg):
    """
    Solve a linear system Ax = b using Gauss-Seidel iteration
    A: coefficient matrix
    b: right-hand side vector
    tol: tolerance
    alg: algorithm to use
    """ 
    
    # set max iterations
    max_iter = 10000

    # Check that A is square and compatible with b
    if A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]:
        raise ValueError("Dimensions of A and b are incompatible or A is not square.")
    # Copy A to avoid altering the original matrix
    A = A.copy()
    # Initialize solution vector x
    x = np.zeros_like(b, dtype=np.double)

    if alg == 'seidel':
        # Gauss-Seidel Iteration
        for k in range(max_iter):
            x_old = x.copy()
            # Loop over rows for the Gauss-Seidel update
            for i in range(A.shape[0]):
                sum1 = np.dot(A[i, :i], x[:i])
                sum2 = np.dot(A[i, (i + 1):], x_old[(i + 1):])
                x[i] = (b[i] - sum1 - sum2) / A[i, i]

            # Convergence check
            if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tol:
                return x
        raise RuntimeWarning("Solution did not converge after maximum number of iterations.")    

    elif alg == 'jacobi':
        # Jacobi Iteration
        T = A - np.diag(np.diagonal(A))
        for k in range(max_iter):
            x_old = x.copy()
            x[:] = (b - np.dot(T, x)) / np.diagonal(A)
            if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tol:
                return x
        raise RuntimeWarning("Solution did not converge after maximum number of iterations.")
        
    else:
        print('Invalid algorithm. Choose either "seidel" or "jacobi".')
        return None

def spline_function(xd,yd,order):
       