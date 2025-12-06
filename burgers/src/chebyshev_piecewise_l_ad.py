import numpy as np
from scipy.optimize import minimize
from util.chebyshev_diff import chebyshev_diff_matrix
from chebyshev_piecewise_ad import fixed_l_inference_system

def construct_loss(n_y, D1_4, D2_4):
    """
    Construct loss function for lambda inference.
    
    The loss function evaluates the absolute difference in the fourth derivative
    at the interface y=0 between the two segments. This measures the discontinuity
    of the fourth derivative, which should be zero for smooth solutions.
    
    Parameters
    ----------
    n_y : int
        Number of y points
    D1_4 : ndarray
        First row of D^4 matrix for segment 1 (evaluates at y=0 from left)
    D2_4 : ndarray
        Last row of D^4 matrix for segment 2 (evaluates at y=0 from right)
    
    Returns
    -------
    loss : callable
        Function that takes lambda as input and returns the loss value
    """
    n_segment = int(n_y/2)+1
    def loss(l):
        """
        Compute loss as absolute difference of fourth derivative at interface.
        
        Parameters
        ----------
        l : float
            Parameter lambda
        
        Returns
        -------
        loss_value : float
            Absolute difference |D^4 U_1(0) - D^4 U_2(0)|
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
    Infer lambda using nested optimization with Nelder-Mead method.

    This function uses a nested optimization strategy: for each candidate lambda,
    it solves the fixed-lambda problem to obtain the solution, then evaluates
    the loss (fourth derivative jump). The outer optimization uses Nelder-Mead
    to minimize this scalar loss function over lambda.

    Parameters
    ----------
    n_y : int
        Number of y points
    tol : float
        Tolerance for optimization
    method : str
        Method for inner optimization (not used, kept for compatibility)
    maxiter : int
        Maximum number of iterations for Nelder-Mead
    n_x0 : int
        Number of random initializations for Nelder-Mead

    Returns
    -------
    best_l : float
        Optimal lambda value that minimizes the loss
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
if __name__ == "__main__":
    n_y = 64
    tol = 1e-10
    method = 'lbfgs'
    maxiter = 50000

    n_x0 = 20

    l = l_inference_system(n_y, tol=tol, method=method, maxiter=maxiter, n_x0=n_x0)

    print(f"(n_y={n_y}): "
            f"l={l}")