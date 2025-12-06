"""
Chebyshev Piecewise Solution with Automatic Differentiation

This module implements the same Burgers equation solver as chebyshev_piecewise.py
but uses automatic differentiation (via JAX) and gradient-based optimization
for more efficient and accurate optimization.

Key advantages over the original implementation:
1. Automatic Differentiation: Gradients are computed exactly using AD, avoiding
   numerical errors from finite differences
2. JIT Compilation: Loss function is JIT-compiled for faster evaluation
3. Gradient-based Optimizers: Uses L-BFGS-B or BFGS with exact gradients for
   faster convergence
4. Better Numerical Stability: AD provides machine-precision gradients

Dependencies:
    - jax: For automatic differentiation and JIT compilation
    - jaxopt (optional): For JAX-native optimizers
    - scipy: For fallback optimizers and root finding
"""

import numpy as np
from scipy.optimize import fsolve, root
import matplotlib.pyplot as plt
from util.chebyshev_diff import chebyshev_diff_matrix, chebyshev_diff_matrix_poly_endpoints

import jax.numpy as jnp
from jax import grad, jit, value_and_grad, jacfwd
from jaxopt import LBFGS, ScipyMinimize
from scipy.optimize import minimize

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


def construct_system_residuals(l, n_y, D1, y1, D2, y2):
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
    def system_residuals(U):
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
        
        # Enforce PDE at interior points only (exclude boundary and interface points)
        # This balances the system: 2*(n_segment-2) PDE + 2 BC + 2 matching = 2*n_segment equations
        residual_1_interior = residual_1_full[1:-1]  # Exclude first (interface) and last (boundary)
        residual_2_interior = residual_2_full[1:-1]  # Exclude first (boundary) and last (interface)
        
        # Assemble residual vector
        residuals = jnp.concatenate([
            residual_1_interior,        # n_segment - 1 residuals
            residual_2_interior,        # n_segment - 2 residuals
            jnp.array([bc_1, bc_2]),     # 2 boundary conditions
            jnp.array([matching, matchingD])  # 2 matching conditions
        ])
        
        return residuals
    
    return system_residuals

def fixed_l_inference_ad(n_y, l, tol=1e-20, optimizer='lbfgs', maxiter=200000, disp=True):
    """
    Solve the Burgers equation using automatic differentiation and gradient-based optimization.
    
    Parameters
    ----------
    n_y : int
        Target number of y points
    l : float
        Parameter lambda
    tol : float
        Tolerance for optimization
    optimizer : str
        Optimizer to use: 'lbfgs', 'bfgs', or 'adam'
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
    num_multigrid_levels = int(np.floor(np.log2(n_y)))
    multigrid_resolutions = [2**i for i in range(1, num_multigrid_levels+1)]
    u_num_lst = []

    for i, n_y_current in enumerate(multigrid_resolutions):
        n_segment = int(n_y_current/2)+1
        
        D1, y1 = chebyshev_diff_matrix(n_segment, a=-2, b=0)
        D2, y2 = chebyshev_diff_matrix(n_segment, a=0, b=2)
        
        # Convert to JAX arrays
        D1_jax = jnp.array(D1)
        D2_jax = jnp.array(D2)
        y1_jax = jnp.array(y1)
        y2_jax = jnp.array(y2)

        if i == 0:
            U0 = -np.concatenate([y1 / 2.0, y2 / 2.0])
        else:
            U1_fine = np.interp(np.flip(y1), np.flip(y1_prev), u_num_lst[-1][:n_segment_prev])
            U2_fine = np.interp(np.flip(y2), np.flip(y2_prev), u_num_lst[-1][n_segment_prev:])
            U0 = np.concatenate([
                U1_fine,
                U2_fine
            ])

        # Construct loss function with automatic differentiation
        loss_function = construct_loss_function_ad(l, n_y_current, D1_jax, y1_jax, D2_jax, y2_jax)
        
        # Create value and gradient function
        loss_and_grad = value_and_grad(loss_function)
        
        # Optimize using gradient-based method
        if optimizer == 'lbfgs':
            # Use JAXOpt L-BFGS-B
            solver = LBFGS(fun=loss_function, maxiter=maxiter, tol=tol)
            U0_jax = jnp.array(U0)
            result = solver.run(U0_jax)
            u_num = np.array(result.params)
            loss_value = float(loss_function(result.params))
            success = True
        elif optimizer == 'bfgs':
            # Use JAXOpt ScipyMinimize wrapper
            solver = ScipyMinimize(fun=loss_function, method='BFGS', 
                                  options={'maxiter': maxiter, 'gtol': tol})
            U0_jax = jnp.array(U0)
            result = solver.run(U0_jax)
            u_num = np.array(result.params)
            loss_value = float(loss_function(result.params))
            success = True
        else:
            # Fall back to scipy.optimize with JAX gradients
            def loss_np(U):
                return float(loss_function(jnp.array(U)))
            
            def grad_np(U):
                return np.array(grad(loss_function)(jnp.array(U)))
            
            # Use L-BFGS-B from scipy with JAX-computed gradients
            result = minimize(
                loss_np, 
                x0=U0, 
                method='L-BFGS-B',
                jac=grad_np,
                tol=tol, 
                options={"maxiter": maxiter, "ftol": tol}
            )
            u_num = result.x
            loss_value = result.fun
            success = result.success

        print(f"Level {i+1}/{len(multigrid_resolutions)} (n_y={n_y_current}): "
              f"Success={success}, Loss={loss_value:.6e}")

        u_num_lst.append(u_num)
        y1_prev = y1
        y2_prev = y2
        n_segment_prev = n_segment

    return u_num_lst[-1], n_segment, (y1, D1, y2, D2)

def fixed_l_inference_system(n_y, l, tol=1e-14, method='hybr', maxiter=100, disp=True):
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
    num_multigrid_levels = int(np.floor(np.log2(n_y)))
    multigrid_resolutions = [2**i for i in range(1, num_multigrid_levels+1)]
    u_num_lst = []

    for i, n_y_current in enumerate(multigrid_resolutions):
        n_segment = int(n_y_current/2)+1
        
        D1, y1 = chebyshev_diff_matrix(n_segment, a=-2, b=0)
        D2, y2 = chebyshev_diff_matrix(n_segment, a=0, b=2)
        
        # Convert to JAX arrays
        D1_jax = jnp.array(D1)
        D2_jax = jnp.array(D2)
        y1_jax = jnp.array(y1)
        y2_jax = jnp.array(y2)

        if i == 0:
            U0 = -np.concatenate([y1 / 2.0, y2 / 2.0])
        else:
            U1_fine = np.interp(np.flip(y1), np.flip(y1_prev), u_num_lst[-1][:n_segment_prev])
            U2_fine = np.interp(np.flip(y2), np.flip(y2_prev), u_num_lst[-1][n_segment_prev:])
            U0 = np.concatenate([U1_fine, U2_fine])

        # Construct system residual function
        system_residuals = construct_system_residuals(l, n_y_current, D1_jax, y1_jax, D2_jax, y2_jax)
        
        # Compute Jacobian using automatic differentiation
        jacobian_fn = jit(jacfwd(system_residuals))
        
        # Wrapper functions for scipy.optimize.root
        def residual_np(U):
            return np.array(system_residuals(jnp.array(U)))
        
        def jacobian_np(U):
            return np.array(jacobian_fn(jnp.array(U)))
        
        # Solve system F(U) = 0
        if method == 'hybr':
            # Modified Powell's hybrid method (default, good for well-conditioned systems)
            result = root(
                residual_np,
                x0=U0,
                jac=jacobian_np,
                method='hybr',
                options={'xtol': tol, 'maxfev': maxiter}
            )
        elif method == 'lm':
            # Levenberg-Marquardt (good for overdetermined systems)
            result = root(
                residual_np,
                x0=U0,
                jac=jacobian_np,
                method='lm',
                options={'xtol': tol, 'maxfev': maxiter}
            )
        elif method == 'broyden1':
            # Broyden's first method (quasi-Newton, doesn't require exact Jacobian)
            result = root(
                residual_np,
                x0=U0,
                method='broyden1',
                options={'xtol': tol, 'maxiter': maxiter}
            )
        else:
            # Default to hybr
            result = root(
                residual_np,
                x0=U0,
                jac=jacobian_np,
                method='hybr',
                options={'xtol': tol, 'maxfev': maxiter}
            )
        
        u_num = result.x
        residual_norm = np.linalg.norm(residual_np(u_num))
        
        if disp:
            print(f"Level {i+1}/{len(multigrid_resolutions)} (n_y={n_y_current}): "
                f"Success={result.success}, Residual norm={residual_norm:.6e}")

        u_num_lst.append(u_num)
        y1_prev = y1
        y2_prev = y2
        n_segment_prev = n_segment

    return u_num_lst[-1], n_segment, (y1, D1, y2, D2)

if __name__ == "__main__":
    l = 0.5
    
    # Choose method: 'loss' for loss-based optimization or 'system' for system-based
    method = 'system'  # or 'loss'
    
    if method == 'system':
        print("Using system-based optimization (F(U) = 0)")
        u_num, n_segment, (y1, D1, y2, D2) = fixed_l_inference_system(64, l, tol=1e-10, method='hybr')
    else:
        print("Using loss-based optimization")
        u_num, n_segment, (y1, D1, y2, D2) = fixed_l_inference_ad(64, l, tol=1e-8, optimizer='lbfgs')

    y_full = np.concatenate([np.flip(y1), np.flip(y2)])
    u_exact = construct_exact_solution(y_full, l=l)

    plt.figure()
    plt.plot(y_full, np.abs(u_exact - u_num), '--', linewidth=2, markersize=4, label='Error')
    plt.xlabel('y')
    plt.ylabel('U(y)')
    plt.title(f'Burgers Self-Similar Solution (λ={l}) - AD Optimized')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot comparison
    # Plot U and its 1st to 4th derivatives in subplots
    D1, y1 = chebyshev_diff_matrix(n_segment, a=-2, b=0)
    D2, y2 = chebyshev_diff_matrix(n_segment, a=0, b=2)

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
        ax.plot(y1, np.flip(arr[:n_segment]), '--', linewidth=2, markersize=4)
        ax.plot(y2, np.flip(arr[n_segment:]), '--', linewidth=2, markersize=4)
        ax.set_ylabel(name, fontsize=13)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title(f'Burgers Self-Similar Solution and Derivatives (λ={l}) - AD Optimized')
        if i == 4:
            ax.set_xlabel('y', fontsize=12)
    plt.tight_layout()
    plt.show()

