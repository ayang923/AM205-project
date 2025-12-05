from typing import Any


import numpy as np
import matplotlib.pyplot as plt
from chebyshev_piecewise import fixed_l_inference, construct_loss_function
from util.chebyshev_diff import chebyshev_diff_matrix
from scipy.optimize import minimize, dual_annealing
from scipy.optimize import fsolve

def verbose_callback(x, f, context):
    # context: 0 = normal iteration, 1/2 = before/after local search
    print(f"context={context}, f={f:.6e}, lambda={x[-1]}")
    # return False to continue, True to stop early
    return False

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
    n_y = 64

    n_segment = int(n_y/2)+1
    D1, y1 = chebyshev_diff_matrix(n_segment, a=-2, b=0)
    D2, y2 = chebyshev_diff_matrix(n_segment, a=0, b=2)

    l_values = [0.45] #np.random.uniform(0.3, 0.7, 1)
    u_num_lst = []
    for i, l in enumerate[Any](l_values):
        u_num, _, _ = fixed_l_inference(64, l, tol=1e-10)
        u_num_lst.append(u_num)

    def loss_function(params):
        U = params[:n_y+2]
        l = params[-1]

        U1 = np.flip(U[:n_segment])
        U2 = np.flip(U[n_segment:])


        D4U1 = np.linalg.matrix_power(D1, 4)@U1
        D4U2 = np.linalg.matrix_power(D2, 4)@U2

        D4matching = D4U1[-1] - D4U2[0]

        return construct_loss_function(l, n_y)(U) + 1e-10*(D4matching)**2 + 1e-15*1/n_y*np.sum(D4U1**2+D4U2**2)

    print("Starting Variable Inference")
    x0 = np.concatenate((u_num_lst[-1], [l_values[-1]]))

    u_exact = construct_exact_solution(np.concatenate([np.flip(y1), np.flip(y2)]), l=0.5)
    print(loss_function(np.concatenate((u_exact, [0.5]))))

    bounds = [(-1.5, 1.5)] * (n_y+2) + [(0.3, 0.7)]
    result_annealing = dual_annealing(loss_function, x0=x0,bounds=bounds, maxiter=1000, callback=verbose_callback)

    print(result_annealing)
    result = minimize(loss_function, x0=x0, bounds=bounds, method='SLSQP', tol=1e-14, options={"maxiter": 1000})

    print(result.success)
    print(result.fun)
    print(result.x[-1])
    # print(result)
    # plt.figure()
    # plt.plot(l_values, u_num_lst, '--', linewidth=2, markersize=4)
    # plt.xlabel('l')
    # plt.ylabel('u_num')
    # plt.title('Varying l Inference')
    # plt.show()