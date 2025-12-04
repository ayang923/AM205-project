import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from util.chebyshev_diff import chebyshev_diff_matrix, chebyshev_diff_matrix_poly_endpoints
from fixed_l_inference import fixed_l_inference

if __name__ == "__main__":
    n_y = 256

    
    l = 0.7
    U0 = fixed_l_inference(n_y, l)

    D, y = chebyshev_diff_matrix(n_y+1)
    D3 = D@D@D
    diff_reg_msk = (y >= -0.5) & (y <= 0.5)
    def loss_function(params):
        U = params[:-1]
        l = params[-1]

        residual = (-l*U + ((1+l)*y + U)*(D@U))
        residual_diff = (D3@U)[diff_reg_msk]

        plt.plot(residual_diff, '--')
        plt.show()
        return 1/n_y * np.sum(residual[1:-1]**2) + 1/2*((U[-1]-1)**2 + (U[0]+1)**2) + 1/np.sum(diff_reg_msk)*np.sum(residual_diff**2)
    
    loss_function(np.concatenate((U0, [l])))
    x0 = np.concatenate((U0, [l]))
    bounds = [(None, None)] * len(U0) + [(0.3, 0.7)]


    # result = minimize(loss_function, x0=x0, bounds=bounds, method='SLSQP', tol=1e-10, options={"maxiter": 10000})

    # u_num = result.x[:-1]
    # l = result.x[-1]

    print(u_num)
    plt.tight_layout()
    plt.show()