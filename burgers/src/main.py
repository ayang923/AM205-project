import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from chebyshev_diff_matrix import chebyshev_diff_matrix
from scipy.interpolate import interpn


def refine_array(arr):
    """
    Refine an array by a factor of 2 by inserting averages between adjacent points.
    
    For an input array of length N, returns an array of length 2N-1 where:
    - Original points are preserved
    - New points are inserted as averages of adjacent original points
    
    Parameters
    ----------
    arr : ndarray
        Input array of values
    
    Returns
    -------
    refined : ndarray
        Refined array with 2N-1 points
    
    Examples
    --------
    >>> refine_array([1, 3, 5])
    array([1., 2., 3., 4., 5.])
    """
    arr = np.asarray(arr)
    if arr.size < 2:
        return arr
    
    # Create output array with 2N-1 points
    n = arr.size
    refined = np.zeros(2 * n - 1)
    
    # Keep original points at even indices
    refined[::2] = arr
    
    # Insert averages at odd indices
    refined[1::2] = (arr[:-1] + arr[1:]) / 2.0
    
    return refined

def exact_solution_zero(U, y, l=0.4):
    return -U-U**(1+1/l)-y

def construct_exact_solution(y_mesh, l=0.4):
    # Constructs exact solution
    u_exact = np.zeros(y_mesh.size)

    n_y_mesh = y_mesh[y_mesh < 0]
    n_indices = np.where(y_mesh < 0)[0]
    for idx, (i, y) in enumerate(zip(n_indices, n_y_mesh)):
        ig = 1 if idx == 0 else u_exact[n_indices[idx-1]]
        u_exact[i] = fsolve(lambda U: exact_solution_zero(U, y, l=l), ig)[0]

    p_indices = np.where(y_mesh > 0)[0]
    u_exact[p_indices] = -u_exact[n_indices][::-1]

    return u_exact

if __name__ == "__main__":
    l = 0.4

    multigrid_resolutions = [3, 5, 9, 17, 33, 65, 129, 257, 513]
    multigrid_results = []
    for i, n_y in enumerate(multigrid_resolutions):
        D, y = chebyshev_diff_matrix(n_y)
        def loss_function(U):
            return 1/n_y * np.sum((-l*U + ((1+l)*y + U)*(D@U))[1:-1]**2) + 1/2*((U[-1]-1)**2 + (U[0]+1)**2)
        
        U0 = -y / 2.0 if i == 0 else refine_array(multigrid_results[i-1])
        result = minimize(loss_function, x0=U0, method='SLSQP', tol=1e-8, options={"maxiter": 10000})

        multigrid_results.append(result.x)
        y_prev = y

    # Construct exact solution on Chebyshev points for comparison
    # u_exact = construct_exact_solution(y, l=l)
    u_num = multigrid_results[-1]

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(y, u_num, 'o-', label='Numerical (Chebyshev)', markersize=4, linewidth=2)
    # plt.plot(y, u_exact, '--', label='Exact', linewidth=2)
    plt.xlabel('y')
    plt.ylabel('U(y)')
    plt.title(f'Burgers Self-Similar Solution (λ={l})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

