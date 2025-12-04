import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import savgol_filter

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

def evaluate_u(coeffs, y):
    return np.sum(coeffs * np.cos(np.pi * y / 2.0))

if __name__ == "__main__":
    l = 0.4
    
    n_coeffs = 1
    n_y = 32
    y = np.linspace(-2, 2, n_y)
    dy = 4/n_y

    n_coeffs_grid = [2, 4, 8, 16, 32]
    results = []
    for idx, n_coeffs in enumerate(n_coeffs_grid):
        print(n_coeffs)
        eval_mat = np.zeros((n_y, n_coeffs))
        for i in range(n_coeffs):
            eval_mat[:, i] = np.sin(np.pi * y / 2.0 * (i+1))

        D_eval_mat = np.zeros((n_y, n_coeffs))
        for i in range(n_coeffs):
            D_eval_mat[:, i] = np.pi * (i+1) / 2.0 * np.cos(np.pi * y / 2.0 * (i+1))


        # def loss_function(coeffs):
        #     U = eval_mat@coeffs
        #     DU = D_eval_mat@coeffs
        #     return 1/n_y * np.sum((-l*U + ((1+l)*y + U)*DU)[1:-1]**2) + 1/2*((U[-1]-1)**2 + (U[0]+1)**2)
        def loss_function(coeffs):
            U = eval_mat@coeffs
            DU = D_eval_mat@coeffs
            return 1/2*((U[-1]-1)**2 + (U[0]+1)**2)

        print(loss_function(np.array([-2/np.pi, 1/np.pi]).reshape(-1, 1)))

        # print(idx)
        # x0 = [-2/np.pi, 1/np.pi] if idx == 0 else results[idx-1]
        # print(x0)
        # result = minimize(loss_function, x0=np.random.rand(n_coeffs), method='SLSQP', tol=1e-10, options={"maxiter": 10000})

        # print(result)
        # results.append(result.x)
    # Construct exact solution on Chebyshev points for comparison
    # u_exact = construct_exact_solution(y, l=l)
    print(result)

    U = eval_mat@result.x
    DU = D_eval_mat@result.x
    plt.plot(y, U, '--')
    plt.plot(y, DU, '--')
    plt.show()

    # Plot comparison
    # Plot U and its 1st to 4th derivatives in subplots
    # derivatives = [u_num]
    # label_names = ["$U$", "$U'$", "$U''$", "$U'''$", "$U^{(4)}$"]

    # current = u_num.copy()
    # for i in range(4):
    #     current = savgol_filter(current, window_length=11, polyorder=3, deriv=i+1, delta=dy)
    #     derivatives.append(current.copy())

    # fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    # for i, (ax, arr, name) in enumerate(zip(axs, derivatives, label_names)):
    #     ax.plot(y, arr, '--', linewidth=2, markersize=4)
    #     ax.set_ylabel(name, fontsize=13)
    #     ax.grid(alpha=0.3)
    #     if i == 0:
    #         ax.set_title(f'Burgers Self-Similar Solution and Derivatives (λ={l})')
    #     if i == 4:
    #         ax.set_xlabel('y', fontsize=12)
    # plt.tight_layout()
    # plt.show()