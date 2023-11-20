import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline


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
    """
    Compute the spline function for a given set of data points
    xd: float array, increasing values of x 
    yd: float array, corresponding values of y = f(x)
    order: order of the spline function (1, 2, 3) default is 3
    """
    
    import numpy as np
    from scipy.interpolate import interp1d, UnivariateSpline

    def spline_function(xd, yd, order):
        # Validation checks
        # Converting to numpy arrays
        xd = np.asarray(xd).flatten()
        yd = np.asarray(yd).flatten()

        if xd.size != yd.size:
            raise ValueError("xd and yd must have the same length.")
        if len(np.unique(xd)) != xd.size:
            raise ValueError("xd contains repeated values.")
        if not np.array_equal(xd, np.sort(xd)):
            raise ValueError("xd values must be in increasing order.")
        if order not in [1, 2, 3]:
            raise ValueError("order must be 1, 2, or 3.")

        # Creating the spline
        if order == 1:
            spline = interp1d(xd, yd, kind='linear')
        elif order == 2:
            spline = interp1d(xd, yd, kind='quadratic')
        elif order == 3:
            spline = interp1d(xd, yd, kind='cubic' )
        
        # Check if input is outside the range of xd values
        if np.any(xd[0] > xd):
            raise ValueError("x is outside the range of xd values.", xd,'is less than', xd[0])
        if np.any(xd > xd[-1]):
            raise ValueError("x is outside the range of xd values.", xd,'is greater than', xd[-1])    
        
        return spline
    