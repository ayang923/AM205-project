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

def construct_system_residuals(n_y, D1, y1, D2, y2, D1_4, D2_4):
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
    
    @jit
    def system_residuals(params):
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
        U = params[:2*n_segment]
        l = params[-1]
        U1 = jnp.flip(U[:n_segment])
        U2 = jnp.flip(U[n_segment:])

        DU1 = D1 @ U1
        DU2 = D2 @ U2
    
        # PDE residuals at all points
        residual_1_full = -l*U1 + ((1+l)*y1 + U1)*DU1
        residual_2_full = -l*U2 + ((1+l)*y2 + U2)*DU2

        # Boundary conditions
        bc_1 = U1[-1] - 1.0  # U(-2) = 1
        bc_2 = U2[0] + 1.0    # U(2) = -1

        # Matching conditions at interface (y=0)
        matching = U1[0] - U2[-1]      # Continuity of U
        matchingD = DU1[0] - DU2[-1]   # Continuity of U'

        matchingD4 = D1_4 @ U1 - D2_4 @ U2   # Continuity of U''
        
        # Enforce PDE at interior points only (exclude boundary and interface points)
        # This balances the system: 2*(n_segment-2) PDE + 2 BC + 2 matching = 2*n_segment equations
        residual_1_interior = residual_1_full[1:-1]  # Exclude first (interface) and last (boundary)
        residual_2_interior = residual_2_full[1:-1]  # Exclude first (boundary) and last (interface)
        
        # Assemble residual vector
        residuals = jnp.concatenate([
            residual_1_interior,        # n_segment - 2 residuals
            residual_2_interior,        # n_segment - 2 residuals
            jnp.array([bc_1, bc_2]),     # 2 boundary conditions
            jnp.array([matching, matchingD]),  # 2 matching conditions
            jnp.array([1e-6*matchingD4])  # 1 matching condition for fourth derivative
        ])
        
        return residuals
    
    return system_residuals

def l_inference_system(n_y, tol=1e-14, method='hybr', maxiter=100):
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
    
    D1, y1 = chebyshev_diff_matrix(n_segment, a=-2, b=0)
    D2, y2 = chebyshev_diff_matrix(n_segment, a=0, b=2)

    D1_4 = np.linalg.matrix_power(D1, 4)[0, :]
    D2_4 = np.linalg.matrix_power(D2, 4)[-1, :]
    
    # Convert to JAX arrays
    D1_jax = jnp.array(D1)
    D2_jax = jnp.array(D2)
    D1_4_jax = jnp.array(D1_4)
    D2_4_jax = jnp.array(D2_4)
    y1_jax = jnp.array(y1)
    y2_jax = jnp.array(y2)

    l_initial = 0.55

    U0, _, _ = fixed_l_inference_system(n_y, l_initial, method=method, tol=tol, maxiter=maxiter)
    x0 = np.concatenate([U0, [l_initial]])

    # Construct system residual function
    system_residuals = construct_system_residuals(n_y, D1_jax, y1_jax, D2_jax, y2_jax, D1_4_jax, D2_4_jax)
    
    # Compute Jacobian using automatic differentiation
    jacobian_fn = jit(jacfwd(system_residuals))
    
    # Wrapper functions for scipy.optimize.root
    def residual_np(params):
        return np.array(system_residuals(jnp.array(params)))
    
    def jacobian_np(params):
        return np.array(jacobian_fn(jnp.array(params)))
    
    # Solve system F(U) = 0
    if method == 'hybr':
        # Modified Powell's hybrid method (default, good for well-conditioned systems)
        result = root(
            residual_np,
            x0=x0,
            jac=jacobian_np,
            method='hybr',
            options={'xtol': tol, 'maxfev': maxiter}
        )
    elif method == 'lm':
        # Levenberg-Marquardt (good for overdetermined systems)
        result = root(
            residual_np,
            x0=x0,
            jac=jacobian_np,
            method='lm',
            options={'xtol': tol, 'maxfev': maxiter}
        )
    elif method == 'broyden1':
        # Broyden's first method (quasi-Newton, doesn't require exact Jacobian)
        result = root(
            residual_np,
            x0=x0,
            method='broyden1',
            options={'xtol': tol, 'maxiter': maxiter}
        )
    else:
        # Default to hybr
        result = root(
            residual_np,
            x0=x0,
            jac=jacobian_np,
            method='hybr',
            options={'xtol': tol, 'maxfev': maxiter}
        )
    
    u_num = result.x
    residual_norm = np.linalg.norm(residual_np(u_num))

    # Compute condition number at final solution
    J_final = jacobian_np(u_num)
    cond_final = np.linalg.cond(J_final)
    print(f"Condition number at final solution: {cond_final:.2e}")

    
    print(f"(n_y={n_y}): "
            f"Success={result.success}, Residual norm={residual_norm:.6e}")


    return u_num[:-1], u_num[-1], n_segment, (y1, D1, y2, D2)

if __name__ == "__main__":
    n_y = 64
    tol = 1e-10
    method = 'lbfgs'
    maxiter = 50000

    u_num, l, n_segment, (y1, D1, y2, D2) = l_inference_system(n_y, tol=tol, method=method, maxiter=maxiter)

    print(f"(n_y={n_y}): "
            f"l={l}")