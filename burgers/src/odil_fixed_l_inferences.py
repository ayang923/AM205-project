import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import savgol_filter

import jax.numpy as jnp
from jax import grad, jacfwd

from scipy.interpolate import CubicSpline

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

def fixed_l_inference(y, l):
    dy = y[1:]-y[:-1]

    def loss_function(U):
        Uph = (U[0:-1] + U[1:]) / 2.0
        yph = (y[0:-1] + y[1:]) / 2.0
        f = -l*Uph + ((1+l)*yph + Uph)*(U[1:]-U[0:-1])/dy

        return 1/n_y * np.sum(f**2) + 1/2*(U[0]-1)**2 + (U[-1]+1)**2
    
    U0 = -y / 2.0
    result = minimize(loss_function, x0=U0, method='SLSQP', tol=1e-10, options={"maxiter": 10000})

    return result.x, result.fun, result.success

def plot_d4_vs_l(y, l_grid):
    y_msk = (y >= -0.5) & (y <= 0.5)
    d4_grid = np.zeros(len(l_grid))
    for i, l in enumerate(l_grid):
        u_num, fun, success = fixed_l_inference(y, l)
        u_num_func = CubicSpline(y, u_num)
        u_num_d4 = u_num_func.derivative(4)(y)
        d4_grid[i] = np.sum(u_num_d4[y_msk]**2)

        plt.figure()
        plt.plot(y, u_num_d4, '--')
        plt.title(f'λ={l}')
        plt.show()
    
    plt.figure()
    plt.plot(l_grid, d4_grid)
    plt.show()


if __name__ == "__main__":
    l_grid = np.linspace(0.3, 0.7, 41)
    n_y = 128
    s = np.linspace(-1, 1, n_y+1)
    ds = 1/n_y
    p = 3
    y = 2*np.sign(s)*np.abs(s)**p
    dy = y[1:]-y[:-1]
    # y = np.linspace(-2, 2, n_y+1)
    plot_d4_vs_l(y, l_grid)
    # l = 0.7
    # n_y = 128
    # s = np.linspace(-1, 1, n_y+1)
    # ds = 1/n_y
    # p = 4
    # y = 2*np.sign(s)*np.abs(s)**p
    # dy = y[1:]-y[:-1]

    # u_num, fun, success = fixed_l_inference(y, l)

    # plt.figure()
    # plt.plot(y, u_num, '-x')
    # plt.show()

    # # def loss_function(U):
    # #     Uph = (U[0:-1] + U[1:]) / 2.0
    # #     yph = (y[0:-1] + y[1:]) / 2.0
    # #     f = -l*Uph + ((1+l)*yph + Uph)*(U[1:]-U[0:-1])/dy

    # #     # Compute second to fourth finite difference derivatives (central differences)
    # #     U2 = CubicSpline(y, U).derivative()
    # #     reg = 1/n_y*(
    # #         np.sum(U2[1:-1]**2) +
    # #         np.sum(U3**2) +
    # #         np.sum(U4**2)
    # #     )
    # #     # Add regularization to the loss (scale factor 1e-4 to not dominate)
    # #     return 1/n_y * np.sum(f**2) + 1/2*(U[0]-1)**2 + (U[-1]+1)**2 + 1e-4*reg
        
    # # # Reasonable parameters for SLSQP optimizer
    # # result = minimize(
    # #     loss_function, 
    # #     x0=u_num, 
    # #     method='L-BFGS-B',
    # #     tol=1e-10,
    # #     options={"maxiter": 1000000}
    # # )
    # # print(result.success)
    # # print(result.fun)

    # # u_num_smooth = result.x
    # # def loss_function(U):
    # #     Uph = (U[0:-1] + U[1:]) / 2.0
    # #     yph = (y[0:-1] + y[1:]) / 2.0
    # #     f = -l*Uph + ((1+l)*yph + Uph)*(U[1:]-U[0:-1])/dy
    # #     plt.plot(yph, f)
    # #     plt.show()
    # #     # Add regularization to the loss (scale factor 1e-4 to not dominate)
    # #     return 1/n_y * np.sum(f**2) + 1/2*((U[0]-1)**2 + (U[-1]+1)**2)

    # u_exact = construct_exact_solution(y, l=l)
    # # plt.plot(y, u_exact, '--', linewidth=2, markersize=4)
    # # plt.plot(y, u_num, 'o-', linewidth=2, markersize=4)
    # # plt.show()
    # # print(loss_function(u_num))

    # y_msk = (y >= -0.5) & (y <= 0.5)
    # u_num_d = np.diff(u_num)/ds
    # u_num_d2 = np.diff(u_num_d)/ds
    # u_num_d3 = np.diff(u_num_d2)/ds
    # u_num_d4 = np.diff(u_num_d3)/ds


    # print(1e-9*np.sum(u_num_d4[y_msk[4:]]**2))


    # # # Plot comparison
    # # Plot U and its 1st to 4th derivatives in subplots
    # derivatives = [u_num]
    # label_names = ["$U$", "$U'$", "$U''$", "$U'''$", "$U^{(4)}$"]

    # current = u_num.copy()
    # for i in range(4):
    #     current = np.diff(current)/ds
    #     derivatives.append(current.copy())

    # fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    # for i, (ax, arr, name) in enumerate(zip(axs, derivatives, label_names)):
    #     ax.plot(arr, '--', linewidth=2, markersize=4)
    #     ax.set_ylabel(name, fontsize=13)
    #     ax.grid(alpha=0.3)
    #     if i == 0:
    #         ax.set_title(f'Burgers Self-Similar Solution and Derivatives (λ={l})')
    #     if i == 4:
    #         ax.set_xlabel('y', fontsize=12)
    # plt.tight_layout()
    # plt.show()