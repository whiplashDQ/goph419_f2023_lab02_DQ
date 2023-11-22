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
    

# test spline_function
import numpy as np
from linalg_interp import spline_function
from scipy.interpolate import UnivariateSpline

def test_linear_spline():
    x = np.linspace(-10, 11, 12)
    y = 3 * x + 1  # Linear function
    spline = spline_function(x, y, order=1)
    assert np.allclose(spline(x), y), "Linear spline test failed"


def test_quadratic_data():
    x = np.linspace(-10, 11, 12)
    y = 2*x**2 - 3*x + 1  # Quadratic function
    spline = spline_function(x, y, order=2)
    assert np.allclose(spline(x), y), "Quadratic spline test failed"

def test_cubic_data():
    x = np.linspace(-10, 11, 12)
    y = x**3 - 2*x**2 + 3*x + 1  # Cubic function
    spline = spline_function(x, y, order=3)
    assert np.allclose(spline(x), y), "Cubic spline test failed"

def test_univariate_spline():
    x = np.linspace(-10, 11, 12)
    y = np.exp(x)  # Exponential function
    spline_custom = spline_function(x, y, order=3)
    spline_scipy = UnivariateSpline(x, y, k=3, s=0, ext='raise')
    assert np.allclose(spline_custom(x), spline_scipy(x)), "Comparison with UnivariateSpline failed"

# Run tests
if __name__ == "__main__":
    test_linear_spline()
    print("Linear data test passed")
    test_quadratic_data()
    print("Quadratic data test passed")
    test_cubic_data()
    print("Cubic data test passed")
    test_univariate_spline()
    print("Comparison with UnivariateSpline test passed")
