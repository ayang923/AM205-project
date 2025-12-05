import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from util.chebyshev_diff import chebyshev_diff_matrix, chebyshev_diff_matrix_poly_endpoints

def exact_solution_zero(U, y, l=0.4):
    return -U-U**(1+1/l)-y

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

def construct_loss_function(l, n_y):
    n_segment = int(n_y/2)+1
        
    D1, y1 = chebyshev_diff_matrix(n_segment, a=-2, b=0)
    D2, y2 = chebyshev_diff_matrix(n_segment, a=0, b=2)

    def loss_function(U):
        U1 = np.flip(U[:n_segment])
        U2 = np.flip(U[n_segment:])

        DU1 = D1@U1
        DU2 = D2@U2
    
        residual_1 = -l*U1 + ((1+l)*y1 + U1)*DU1
        residual_2 = -l*U2 + ((1+l)*y2 + U2)*DU2

        bc_1 = U1[-1]-1
        bc_2 = U2[0]+1

        matching = U1[0] - U2[-1]
        matchingD = DU1[0] - DU2[-1]

        return np.sum(residual_1[:-1]**2) + 1/2*bc_1**2 + 1/(n_segment) * np.sum(residual_2[1:]**2) + 1/2*bc_2**2 + 1/2*matching**2 + 1/2*matchingD**2

    return loss_function
    

def fixed_l_inference(n_y, l, tol=1e-20):
    num_multigrid_levels = int(np.floor(np.log2(n_y)))
    multigrid_resolutions = [2**i for i in range(1, num_multigrid_levels+1)]
    u_num_lst = []

    for i, n_y in enumerate(multigrid_resolutions):
        n_segment = int(n_y/2)+1
        
        D1, y1 = chebyshev_diff_matrix(n_segment, a=-2, b=0)
        D2, y2 = chebyshev_diff_matrix(n_segment, a=0, b=2)

        U0 = -np.concatenate([y1 / 2.0, y2 / 2.0]) if i == 0 else np.concatenate([refine_array(u_num_lst[-1][:n_segment]), refine_array(u_num_lst[-1][n_segment:])])

        loss_function = construct_loss_function(l, n_y)
        
        result = minimize(loss_function, x0=U0, method='SLSQP', tol=tol, options={"maxiter": 10000})

        print(result.success)
        print(result.fun)

        u_num_lst.append(result.x)

    return u_num_lst[-1], n_segment, (y1, D1, y2, D2)

if __name__ == "__main__":
    l = 0.49
    u_num, n_segment,(y1, D1, y2, D2) = fixed_l_inference(64, l, tol=1e-20)

    y_full = np.concatenate([np.flip(y1), np.flip(y2)])
    u_exact = construct_exact_solution(y_full, l=l)

    plt.figure()
    plt.plot(y_full, np.abs(u_exact - u_num), '--', linewidth=2, markersize=4, label='Error')
    # plt.plot(y_full, u_num, '-x', linewidth=2, markersize=4, label='Numerical')
    plt.xlabel('y')
    plt.ylabel('U(y)')
    plt.title(f'Burgers Self-Similar Solution (λ={l})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    # Plot comparison
    # Plot U and its 1st to 4th derivatives in subplots
    D1, y1 = chebyshev_diff_matrix(n_segment, a=-2, b=0)
    D2, y2 = chebyshev_diff_matrix(n_segment, a=0, b=2)

    u_num = u_exact

    derivatives = [u_num]
    label_names = ["$U$", "$U'$", "$U''$", "$U'''$", "$U^{(4)}$"]

    U1_current = np.flip(u_num[:n_segment])
    U2_current = np.flip(u_num[n_segment:])
    for i in range(4):
        U1_current = D1 @ U1_current
        U2_current = D2 @ U2_current

        derivatives.append(np.concatenate([np.flip(U1_current), np.flip(U2_current)]))

    fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    for i, (ax, arr, name) in enumerate(zip(axs, derivatives, label_names)):
        ax.plot(y_full, arr, '--', linewidth=2, markersize=4)
        ax.set_ylabel(name, fontsize=13)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title(f'Burgers Self-Similar Solution and Derivatives (λ={l})')
        if i == 4:
            ax.set_xlabel('y', fontsize=12)
    plt.tight_layout()
    plt.show()