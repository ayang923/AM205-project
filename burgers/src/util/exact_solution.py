"""
Exact Solution Construction

This module provides functions to construct the exact solution to the
self-similar Burgers equation.
"""

import numpy as np
from scipy.optimize import fsolve


def construct_exact_solution(y_mesh, l=0.4):
    """
    Construct exact solution to the self-similar Burgers equation.
    
    The exact solution is given implicitly by: y = -U - CU^(1+1/l)
    where C is determined by the boundary conditions.
    
    Parameters
    ----------
    y_mesh : ndarray
        Array of y coordinates where solution is evaluated
    l : float, optional
        Parameter lambda. Default is 0.4.
    
    Returns
    -------
    u_exact : ndarray
        Exact solution values at y_mesh points
    """
    def exact_solution_zero(U, y):
        """
        Zero of the exact solution equation.
        
        Parameters
        ----------
        U : float
            Solution value
        y : float
            Spatial coordinate
        
        Returns
        -------
        residual : float
            Value of -U - U^(1+1/l) - y
        """
        return -U-U**(1+1/l)-y
    u_exact = np.zeros(y_mesh.size)

    n_y_mesh = y_mesh[y_mesh < -1e-15]
    n_indices = np.where(y_mesh < -1e-15)[0]


    for idx, (i, y) in enumerate(zip(n_indices, n_y_mesh)):
        ig = 1 if idx == 0 else u_exact[n_indices[idx-1]]
        
        u_exact[i] = fsolve(lambda U: exact_solution_zero(U, y), ig, xtol=1e-8)[0]

    p_indices = np.where(y_mesh > 1e-15)[0]

    u_exact[p_indices] = -u_exact[n_indices][::-1]
    return u_exact