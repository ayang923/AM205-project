"""
Chebyshev Differentiation

This module constructs the Chebyshev differentiation matrix for functions
defined on the interval [-2, 2] using Chebyshev-Gauss-Lobatto points.
"""

import numpy as np


def chebyshev_points(N):
    """
    Compute Chebyshev-Gauss-Lobatto points on [-2, 2].
    
    Parameters
    ----------
    N : int
        Number of points (including endpoints). N >= 2.
    
    Returns
    -------
    x : ndarray
        Array of Chebyshev points, shape (N,)
    """
    if N < 2:
        raise ValueError("N must be at least 2")
    
    # Chebyshev-Gauss-Lobatto points on [-1, 1]: x_j = cos(pi * j / (N-1))
    # Map to [-2, 2] by scaling: y = 2*x
    j = np.arange(N)
    x_standard = np.cos(np.pi * j / (N - 1))
    x = 2 * x_standard  # Map from [-1, 1] to [-2, 2]
    return x


def chebyshev_diff_matrix(N):
    """
    Construct the Chebyshev differentiation matrix D.
    
    The matrix D is such that if u is a vector of function values at
    Chebyshev points, then D @ u approximates the derivative at those points.
    
    Parameters
    ----------
    N : int
        Number of Chebyshev points (including endpoints). N >= 2.
    
    Returns
    -------
    D : ndarray
        Differentiation matrix, shape (N, N)
    x : ndarray
        Chebyshev points on [-2, 2], shape (N,)
    
    References
    ----------
    Trefethen, L. N. (2000). Spectral Methods in MATLAB. SIAM.
    """
    if N < 2:
        raise ValueError("N must be at least 2")
    
    # Compute Chebyshev points on [-2, 2]
    x = chebyshev_points(N)
    
    # Standard Chebyshev points on [-1, 1] for matrix construction
    j = np.arange(N)
    x_standard = np.cos(np.pi * j / (N - 1))
    
    # Initialize differentiation matrix
    D = np.zeros((N, N))
    
    # Compute c_i values: c_0 = c_N = 2, c_i = 1 otherwise
    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0
    
    # Fill in the differentiation matrix using standard [-1, 1] points
    # Then scale by 1/2 to account for mapping to [-2, 2]
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1)**(i + j)) / (x_standard[i] - x_standard[j])
            elif i == 0:
                D[i, i] = (2 * (N - 1)**2 + 1) / 6.0
            elif i == N - 1:
                D[i, i] = -(2 * (N - 1)**2 + 1) / 6.0
            else:
                D[i, i] = -x_standard[i] / (2 * (1 - x_standard[i]**2))
    
    # Scale by 1/2 to account for the mapping from [-1, 1] to [-2, 2]
    # (chain rule: if y = 2x, then d/dy = (1/2) * d/dx)
    D = D / 2.0
    
    return D, x


def polynomial_diff_weights(x, x0, order=1):
    """
    Compute differentiation weights for polynomial interpolation.
    
    Uses Lagrange polynomial interpolation to compute the derivative
    at point x0 given function values at points x.
    
    Parameters
    ----------
    x : ndarray
        Interpolation points (must be distinct)
    x0 : float
        Point where derivative is evaluated
    order : int
        Order of derivative (1 for first derivative)
    
    Returns
    -------
    weights : ndarray
        Weights such that sum(weights * f(x)) = f^(order)(x0)
    
    Notes
    -----
    Uses the standard formula for differentiation weights of Lagrange
    interpolation polynomials. For x0 = x[i], uses the simpler formula.
    """
    n = len(x)
    weights = np.zeros(n)
    
    if order == 1:
        # Find if x0 is one of the interpolation points
        idx_match = None
        for i in range(n):
            if abs(x0 - x[i]) < 1e-14:
                idx_match = i
                break
        
        if idx_match is not None:
            # x0 is exactly one of the interpolation points (x[i])
            # Use the standard formula for Lagrange polynomial derivative at node
            i = idx_match
            for j in range(n):
                if j == i:
                    # Diagonal term: derivative of L_i(x) at x = x_i
                    sum_term = 0.0
                    for k in range(n):
                        if k != i:
                            sum_term += 1.0 / (x[i] - x[k])
                    weights[j] = sum_term
                else:
                    # Off-diagonal term: derivative of L_j(x) at x = x_i
                    # L_j(x) = product_{k≠j} (x - x_k) / (x_j - x_k)
                    # L_j'(x_i) = (1/(x_j - x_i)) * product_{k≠i,k≠j} (x_i - x_k)/(x_j - x_k)
                    product = 1.0 / (x[j] - x[i])
                    for k in range(n):
                        if k != i and k != j:
                            product *= (x[i] - x[k]) / (x[j] - x[k])
                    weights[j] = product
        else:
            # x0 is not an interpolation point
            # Use formula: L_i'(x0) = L_i(x0) * sum_{k≠i} 1/(x0 - x_k)
            for i in range(n):
                # Compute L_i(x0)
                L_i = 1.0
                for j in range(n):
                    if j != i:
                        L_i *= (x0 - x[j]) / (x[i] - x[j])
                
                # Compute sum
                sum_term = 0.0
                for k in range(n):
                    if k != i:
                        sum_term += 1.0 / (x0 - x[k])
                
                weights[i] = L_i * sum_term
    else:
        raise ValueError(f"Order {order} not implemented. Use order=1.")
    
    return weights


def chebyshev_diff_matrix_poly_endpoints(N, M=6):
    """
    Construct Chebyshev differentiation matrix with polynomial interpolation
    at endpoints.
    
    Uses polynomial interpolation (Lagrange) to compute derivatives at the
    M points nearest each endpoint, while using standard Chebyshev differentiation
    for interior points. This can improve accuracy at boundaries.
    
    Parameters
    ----------
    N : int
        Number of Chebyshev points (including endpoints). N >= 2.
    M : int
        Number of endpoint points to use polynomial interpolation for at each boundary.
        Must satisfy 2 <= M <= (N-1)//2. Default is 6.
    
    Returns
    -------
    D : ndarray
        Differentiation matrix, shape (N, N)
    x : ndarray
        Chebyshev points on [-2, 2], shape (N,)
    
    Notes
    -----
    For the M points at each endpoint, this function uses polynomial interpolation
    based on nearby points to compute derivatives, which can be more accurate than
    the standard Chebyshev differentiation matrix at boundaries.
    """
    if N < 2:
        raise ValueError("N must be at least 2")
    if M < 2:
        raise ValueError("M must be at least 2")
    if 2*M >= N:
        raise ValueError(f"M={M} too large. Need 2*M < N (got N={N})")
    
    # Get Chebyshev points
    x = chebyshev_points(N)
    
    # Get standard Chebyshev differentiation matrix
    D_standard, _ = chebyshev_diff_matrix(N)
    
    # Create new differentiation matrix
    D = D_standard.copy()
    
    # Use polynomial interpolation for left endpoint (first M points)
    # Use more points for interpolation to get better accuracy
    interp_points = min(M + 3, N)  # Use M+3 points for interpolation
    
    for i in range(M):
        # For point i, use nearby points for polynomial interpolation
        # Use points from i to i+interp_points-1
        end_idx = min(i + interp_points, N)
        interp_indices = np.arange(i, end_idx)
        
        if len(interp_indices) >= 2:
            x_interp = x[interp_indices]
            # x0 is x[i], which is the first point in x_interp
            weights = polynomial_diff_weights(x_interp, x[i], order=1)
            
            # Set row i of D using polynomial interpolation weights
            D[i, :] = 0.0
            for j, idx in enumerate(interp_indices):
                D[i, idx] = weights[j]
    
    # Use polynomial interpolation for right endpoint (last M points)
    for i in range(N - M, N):
        # For point i, use nearby points for polynomial interpolation
        # Use points from i-interp_points+1 to i
        start_idx = max(i - interp_points + 1, 0)
        interp_indices = np.arange(start_idx, i + 1)
        
        if len(interp_indices) >= 2:
            x_interp = x[interp_indices]
            # x0 is x[i], which is the last point in x_interp
            weights = polynomial_diff_weights(x_interp, x[i], order=1)
            
            # Set row i of D using polynomial interpolation weights
            D[i, :] = 0.0
            for j, idx in enumerate(interp_indices):
                D[i, idx] = weights[j]
    
    return D, x


def chebyshev_diff_matrix_2d(N):
    """
    Construct the second-order Chebyshev differentiation matrix D2.
    
    Parameters
    ----------
    N : int
        Number of Chebyshev points (including endpoints). N >= 2.
    
    Returns
    -------
    D2 : ndarray
        Second-order differentiation matrix, shape (N, N)
    x : ndarray
        Chebyshev points, shape (N,)
    """
    D, x = chebyshev_diff_matrix(N)
    D2 = D @ D  # Second derivative matrix is D^2
    return D2, x


def test_derivatives():
    """Test the Chebyshev differentiation matrix on various functions."""
    print("=" * 70)
    print("Testing Chebyshev Differentiation Matrix")
    print("=" * 70)
    
    # Test with different numbers of points
    N_values = [8, 16, 32]
    
    for N in N_values:
        print(f"\n{'='*70}")
        print(f"Testing with N = {N} points")
        print(f"{'='*70}")
        
        D, x = chebyshev_diff_matrix(N)
        D2, x2 = chebyshev_diff_matrix_2d(N)
        
        # Verify points are on [-2, 2]
        assert np.allclose(x[0], 2.0), f"First point should be 2.0, got {x[0]}"
        assert np.allclose(x[-1], -2.0), f"Last point should be -2.0, got {x[-1]}"
        print(f"✓ Points correctly span [-2, 2]")
        
        # Test 1: u(x) = x, u'(x) = 1
        print(f"\nTest 1: u(x) = x, u'(x) = 1")
        u = x
        du_exact = np.ones_like(x)
        du_approx = D @ u
        error = np.abs(du_approx - du_exact)
        max_error = np.max(error)
        print(f"  Max error: {max_error:.2e}")
        if max_error < 1e-10:
            print(f"  ✓ PASS (machine precision)")
        else:
            print(f"  ✗ FAIL")
        
        # Test 2: u(x) = x^2, u'(x) = 2x
        print(f"\nTest 2: u(x) = x^2, u'(x) = 2x")
        u = x**2
        du_exact = 2 * x
        du_approx = D @ u
        error = np.abs(du_approx - du_exact)
        max_error = np.max(error)
        print(f"  Max error: {max_error:.2e}")
        if max_error < 1e-10:
            print(f"  ✓ PASS (machine precision)")
        else:
            print(f"  ✗ FAIL")
        
        # Test 3: u(x) = x^3, u'(x) = 3x^2
        print(f"\nTest 3: u(x) = x^3, u'(x) = 3x^2")
        u = x**3
        du_exact = 3 * x**2
        du_approx = D @ u
        error = np.abs(du_approx - du_exact)
        max_error = np.max(error)
        print(f"  Max error: {max_error:.2e}")
        if max_error < 1e-10:
            print(f"  ✓ PASS (machine precision)")
        else:
            print(f"  ✗ FAIL")
        
        # Test 4: u(x) = sin(πx/2), u'(x) = (π/2)cos(πx/2)
        print(f"\nTest 4: u(x) = sin(πx/2), u'(x) = (π/2)cos(πx/2)")
        u = np.sin(np.pi * x / 2)
        du_exact = (np.pi / 2) * np.cos(np.pi * x / 2)
        du_approx = D @ u
        error = np.abs(du_approx - du_exact)
        max_error = np.max(error)
        print(f"  Max error: {max_error:.2e}")
        if max_error < 1e-6:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
        
        # Test 5: u(x) = exp(x), u'(x) = exp(x)
        print(f"\nTest 5: u(x) = exp(x), u'(x) = exp(x)")
        u = np.exp(x)
        du_exact = np.exp(x)
        du_approx = D @ u
        error = np.abs(du_approx - du_exact)
        max_error = np.max(error)
        print(f"  Max error: {max_error:.2e}")
        if max_error < 1e-6:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
        
        # Test 6: Second derivative - u(x) = x^2, u''(x) = 2
        print(f"\nTest 6: u(x) = x^2, u''(x) = 2")
        u = x**2
        d2u_exact = 2 * np.ones_like(x)
        d2u_approx = D2 @ u
        error = np.abs(d2u_approx - d2u_exact)
        max_error = np.max(error)
        print(f"  Max error: {max_error:.2e}")
        if max_error < 1e-8:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
        
        # Test 7: Second derivative - u(x) = sin(πx/2), u''(x) = -(π/2)^2 sin(πx/2)
        print(f"\nTest 7: u(x) = sin(πx/2), u''(x) = -(π/2)^2 sin(πx/2)")
        u = np.sin(np.pi * x / 2)
        d2u_exact = -(np.pi / 2)**2 * np.sin(np.pi * x / 2)
        d2u_approx = D2 @ u
        error = np.abs(d2u_approx - d2u_exact)
        max_error = np.max(error)
        print(f"  Max error: {max_error:.2e}")
        if max_error < 1e-4:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
    
    print(f"\n{'='*70}")
    print("Convergence Test: Error vs. Number of Points")
    print(f"{'='*70}")
    
    # Test convergence for a smooth function
    test_func = lambda x: np.sin(np.pi * x / 2)
    test_deriv = lambda x: (np.pi / 2) * np.cos(np.pi * x / 2)
    
    N_conv = [8, 16, 32, 64, 128]
    errors = []
    
    print(f"\nFunction: u(x) = sin(πx/2)")
    print(f"{'N':<6} {'Max Error':<15} {'Order':<10}")
    print("-" * 35)
    
    for i, N in enumerate(N_conv):
        D, x = chebyshev_diff_matrix(N)
        u = test_func(x)
        du_exact = test_deriv(x)
        du_approx = D @ u
        error = np.max(np.abs(du_approx - du_exact))
        errors.append(error)
        
        if i > 0:
            order = np.log(errors[i-1] / errors[i]) / np.log(N_conv[i] / N_conv[i-1])
            print(f"{N:<6} {error:<15.2e} {order:<10.2f}")
        else:
            print(f"{N:<6} {error:<15.2e} {'-':<10}")
    
    print(f"\n{'='*70}")
    print("All tests completed!")
    print(f"{'='*70}")


def test_poly_endpoints():
    """Test the polynomial endpoint differentiation matrix."""
    print("=" * 70)
    print("Testing Chebyshev Differentiation Matrix with Polynomial Endpoints")
    print("=" * 70)
    
    N = 33
    M = 6
    
    print(f"\nTesting with N={N} points, M={M} polynomial endpoints")
    
    # Get both matrices
    D_standard, x = chebyshev_diff_matrix(N)
    D_poly, x_poly = chebyshev_diff_matrix_poly_endpoints(N, M=M)
    
    assert np.allclose(x, x_poly), "Points should be the same"
    
    # Test on a simple function
    u = x**2
    du_exact = 2 * x
    
    du_standard = D_standard @ u
    du_poly = D_poly @ u
    
    error_standard = np.abs(du_standard - du_exact)
    error_poly = np.abs(du_poly - du_exact)
    
    print(f"\nTest: u(x) = x^2, u'(x) = 2x")
    print(f"Standard matrix - Max error: {np.max(error_standard):.2e}")
    print(f"  Endpoint errors: left={error_standard[0]:.2e}, right={error_standard[-1]:.2e}")
    print(f"Poly endpoint matrix - Max error: {np.max(error_poly):.2e}")
    print(f"  Endpoint errors: left={error_poly[0]:.2e}, right={error_poly[-1]:.2e}")
    
    # Show differences at endpoints
    print(f"\nDifference at first {M} points (left endpoint):")
    for i in range(M):
        print(f"  Point {i}: x={x[i]:.4f}, std_err={error_standard[i]:.2e}, poly_err={error_poly[i]:.2e}")
    
    print(f"\nDifference at last {M} points (right endpoint):")
    for i in range(N-M, N):
        print(f"  Point {i}: x={x[i]:.4f}, std_err={error_standard[i]:.2e}, poly_err={error_poly[i]:.2e}")
    
    # Test on a smooth function
    u_smooth = np.sin(np.pi * x / 2)
    du_smooth_exact = (np.pi / 2) * np.cos(np.pi * x / 2)
    
    du_smooth_standard = D_standard @ u_smooth
    du_smooth_poly = D_poly @ u_smooth
    
    error_smooth_standard = np.abs(du_smooth_standard - du_smooth_exact)
    error_smooth_poly = np.abs(du_smooth_poly - du_smooth_exact)
    
    print(f"\nTest: u(x) = sin(πx/2), u'(x) = (π/2)cos(πx/2)")
    print(f"Standard matrix - Max error: {np.max(error_smooth_standard):.2e}")
    print(f"  Endpoint errors: left={error_smooth_standard[0]:.2e}, right={error_smooth_standard[-1]:.2e}")
    print(f"Poly endpoint matrix - Max error: {np.max(error_smooth_poly):.2e}")
    print(f"  Endpoint errors: left={error_smooth_poly[0]:.2e}, right={error_smooth_poly[-1]:.2e}")
    
    print(f"\n{'='*70}")
    print("Test completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "poly":
        test_poly_endpoints()
    else:
        test_derivatives()

