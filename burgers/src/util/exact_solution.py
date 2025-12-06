import numpy as np
from scipy.optimize import fsolve


def construct_exact_solution(y_mesh, l=0.4):

    def exact_solution_zero(U, y):
        return -U-U**(1+1/l)-y

    # Constructs exact solution
    u_exact = np.zeros(y_mesh.size)

    n_y_mesh = y_mesh[y_mesh < -1e-15]
    n_indices = np.where(y_mesh < -1e-15)[0]


    for idx, (i, y) in enumerate(zip(n_indices, n_y_mesh)):
        ig = 1 if idx == 0 else u_exact[n_indices[idx-1]]
        
        u_exact[i] = fsolve(lambda U: exact_solution_zero(U, y), ig, xtol=1e-8)[0]

    p_indices = np.where(y_mesh > 1e-15)[0]

    u_exact[p_indices] = -u_exact[n_indices][::-1]
    return u_exact