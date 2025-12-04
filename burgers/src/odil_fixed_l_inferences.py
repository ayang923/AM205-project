import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

def fixed_l_inference(n_y, l):
    num_multigrid_levels = int(np.floor(np.log2(n_y)))
    multigrid_resolutions = [2**i for i in range(1, num_multigrid_levels+1)]
    multigrid_results = []

    for i, n_y in enumerate(multigrid_resolutions):
        y = np.linspace(-2, 2, n_y+1)
        dy = 4/n_y

        def loss_function(U):
            Uph = (U[0:-1] + U[1:]) / 2.0
            yph = (y[0:-1] + y[1:]) / 2.0
            f = -l*Uph + ((1+l)*yph + Uph)*(U[1:]-U[0:-1])/dy

            return 1/n_y * np.sum(f**2) + 1/2*(U[0]-1)**2 + (U[-1]+1)**2
        
        U0 = -y / 2.0 if i == 0 else refine_array(multigrid_results[i-1])
        result = minimize(loss_function, x0=U0, method='SLSQP', tol=1e-10, options={"maxiter": 10000})

        multigrid_results.append(result.x)

    return multigrid_results[-1]

if __name__ == "__main__":
    l = 0.45
    n_y = 128
    s = np.linspace(-2, 2, n_y+1)
    p = 3
    y = np.sign(s)*np.abs(s)**p
    dy = y[1:]-y[:-1]

    u_num = fixed_l_inference(n_y, l)

    def loss_function(U):
        Uph = (U[0:-1] + U[1:]) / 2.0
        yph = (y[0:-1] + y[1:]) / 2.0
        f = -l*Uph + ((1+l)*yph + Uph)*(U[1:]-U[0:-1])/dy

        # Compute second to fourth finite difference derivatives (central differences)
        U2 = (U[:-2] - 2*U[1:-1] + U[2:]) / dy**2
        U3 = (U[2:] - 2*U[1:-1] + U[:-2])       # third derivative using forward/backward differences is less standard; keep it simple
        U3 = (U[2:] - 2*U[1:-1] + U[:-2]) / dy**2              # get the "second-difference" at midpoints
        U3 = (U3[1:] - U3[:-1]) / dy                           # third derivative at interior points
        U4 = (U[:-4] - 4*U[1:-3] + 6*U[2:-2] - 4*U[3:-1] + U[4:]) / dy**4

        reg = 1/n_y*(
            np.sum(U2[1:-1]**2) +
            np.sum(U3**2) +
            np.sum(U4**2)
        )
        # Add regularization to the loss (scale factor 1e-4 to not dominate)
        return 1/n_y * np.sum(f**2) + 1/2*(U[0]-1)**2 + (U[-1]+1)**2 + 1e-4*reg
        
    # Reasonable parameters for SLSQP optimizer
    result = minimize(
        loss_function, 
        x0=u_num, 
        method='L-BFGS-B',
        tol=1e-10,
        options={"maxiter": 1000000}
    )
    print(result.success)
    print(result.fun)
    
    # Construct exact solution on Chebyshev points for comparison
    # u_exact = construct_exact_solution(y, l=l)
    u_num_smooth = result.x
    def loss_function(U):
        Uph = (U[0:-1] + U[1:]) / 2.0
        yph = (y[0:-1] + y[1:]) / 2.0
        f = -l*Uph + ((1+l)*yph + Uph)*(U[1:]-U[0:-1])/dy
        plt.plot(yph, f)
        plt.show()
        # Add regularization to the loss (scale factor 1e-4 to not dominate)
        return 1/n_y * np.sum(f**2) + 1/2*((U[0]-1)**2 + (U[-1]+1)**2)

    u_exact = construct_exact_solution(y, l=l)
    plt.plot(y, u_exact, '--', linewidth=2, markersize=4)
    plt.plot(y, u_num, 'o-', linewidth=2, markersize=4)
    plt.show()
    print(loss_function(u_num))


    # Plot comparison
    # Plot U and its 1st to 4th derivatives in subplots
    derivatives = [u_num]
    label_names = ["$U$", "$U'$", "$U''$", "$U'''$", "$U^{(4)}$"]

    current = u_num.copy()
    for i in range(4):
        current = np.diff(current)/dy
        derivatives.append(current.copy())

    fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    for i, (ax, arr, name) in enumerate(zip(axs, derivatives, label_names)):
        ax.plot(arr, '--', linewidth=2, markersize=4)
        ax.set_ylabel(name, fontsize=13)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title(f'Burgers Self-Similar Solution and Derivatives (λ={l})')
        if i == 4:
            ax.set_xlabel('y', fontsize=12)
    plt.tight_layout()
    plt.show()