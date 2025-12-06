from typing import Any

import numpy as np
from scipy.optimize import fsolve, root
import matplotlib.pyplot as plt
from util.chebyshev_diff import chebyshev_diff_matrix
from chebyshev_piecewise_ad import fixed_l_inference_system

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad, jacfwd
from jaxopt import LBFGS, ScipyMinimize
from scipy.optimize import minimize

def construct_loss(n_y, D1_4, D2_4):
    """
    Construct system of equations F(U) = 0 instead of loss function.
    
    The system includes:
    - PDE residuals: -l*U + ((1+l)*y + U)*DU = 0 (at interior points only)
    - Boundary conditions: U1[-1] = 1, U2[0] = -1
    - Matching conditions: U1[0] = U2[-1], DU1[0] = DU2[-1]
    
    The system is balanced: 2*(n_segment-2) PDE equations + 2 BC + 2 matching = 2*n_segment
    equations for 2*n_segment unknowns.
    
    This approach solves the system directly rather than minimizing a weighted loss,
    which can be faster and avoids the need to tune penalty weights.
    
    Parameters
    ----------
    l : float
        Parameter lambda
    n_y : int
        Number of y points
    D1, D2 : jnp.ndarray
        Differentiation matrices for segments 1 and 2
    y1, y2 : jnp.ndarray
        y coordinates for segments 1 and 2
    
    Returns
    -------
    system_residuals : callable
        JAX-compiled function that returns residual vector F(U)
    """
    n_segment = int(n_y/2)+1
    def loss(l):
        """
        Compute residual vector F(U) = 0.
        
        Parameters
        ----------
        U : jnp.ndarray
            Solution vector [U1, U2] concatenated
        
        Returns
        -------
        residuals : jnp.ndarray
            Vector of residuals (should be zero at solution)
            Structure: [residual_1, residual_2, bc_1, bc_2, matching, matchingD]
        """
        u_num = fixed_l_inference_system(n_y, l=l, tol=1e-10, method='hybr', disp=False)[0]
        U1 = np.flip(u_num[:n_segment])
        U2 = np.flip(u_num[n_segment:])

        D4_match = np.abs(D1_4 @ U1 - D2_4 @ U2)   # Continuity of U^4
        if not np.isfinite(D4_match):
            return 1e10

        return D4_match
    return loss

def l_inference_system(n_y, tol=1e-14, method='hybr', maxiter=100, n_x0=20):
    """
    Solve as a system of equations F(U) = 0 using Newton's method.

    This approach directly solves the system of nonlinear equations rather than
    minimizing a loss function. It can be faster and more accurate for well-conditioned
    problems, and avoids the need to tune penalty weights.

    Parameters
    ----------
    n_y : int
        Target number of y points
    l : float
        Parameter lambda
    tol : float
        Tolerance for root finding
    method : str
        Method: 'hybr' (Powell's hybrid), 'lm' (Levenberg-Marquardt), or 'broyden1'
    maxiter : int
        Maximum number of iterations

    Returns
    -------
    u_num : ndarray
        Numerical solution
    n_segment : int
        Number of points per segment
    mesh_info : tuple
        (y1, D1, y2, D2) mesh information
    """

    n_segment = int(n_y/2)+1
    
    D1, _ = chebyshev_diff_matrix(n_segment, a=-2, b=0)
    D2, _ = chebyshev_diff_matrix(n_segment, a=0, b=2)

    D1_4 = np.linalg.matrix_power(D1, 4)[0, :]
    D2_4 = np.linalg.matrix_power(D2, 4)[-1, :]


    # Construct system residual function
    loss = construct_loss(n_y, D1_4, D2_4)
    
    # Only optimize lambda (assume loss function takes l as input)
    best_l = None
    best_loss = np.inf
    for i in range(n_x0):
        x0 = np.random.uniform(0.45, 0.55)
        result = minimize(loss, x0=x0, method='Nelder-Mead', tol=tol, options={'maxiter': maxiter, "disp": True})
        l_opt = result.x
        if result.fun < best_loss:
            best_loss = result.fun
            best_l = l_opt
    
    return best_l
    return l_opt
if __name__ == "__main__":
    n_y = 64
    tol = 1e-10
    method = 'lbfgs'
    maxiter = 50000

    n_x0 = 20

    l = l_inference_system(n_y, tol=tol, method=method, maxiter=maxiter, n_x0=n_x0)

    print(f"(n_y={n_y}): "
            f"l={l}")