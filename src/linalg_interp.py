import numpy as np

air_temp_g = np.loadtxt('air_density_vs_temp_eng_toolbox.txt')
#convert 1st column value from g/m^3 to kg/m^3
air_temp_kg = air_temp_g[:,0] * 1e3
#replace 1st column with kg/m^3
air_vs_temp = np.column_stack((air_temp_kg, air_temp_g[:,1]))
water_vs_temp = np.loadtxt('water_density_vs_temp_usgs.txt')


def gauss_iter_solve(A,b,tol = 1e-8,alg):
    """
    Solve a linear system Ax = b using Gauss-Seidel iteration
    A: coefficient matrix
    b: right-hand side vector
    tol: tolerance
    alg: algorithm to use
    """ 
    ### SET A AND b HERE ###




    # get the size of the matrix
    n = len(A)
    # set max iterations
    max_iter = 10000
    
    A = A.copy()
    
    if alg == 'seidel':
         # Iterate
        for k in range(max_iterations):
            x_old = x.copy()
            # Loop over rows for the Gauss-Seidel update
            for i in range(A.shape[0]):
                x[i] = b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, (i + 1):], x_old[(i + 1):])

            # Convergence check
            if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerance:
                break
        return x


    if alg == 'jacobi':
        for k in range(max_iter):
            x = np.dot(A_norm_s, x_old) + b_norm
            res = np.linalg.norm(x - x_old, ord=np.inf)