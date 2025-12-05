import numpy as np
import matplotlib.pyplot as plt
from chebyshev_piecewise import fixed_l_inference

if __name__ == "__main__":
    n_y = 64

    l_values = np.random.uniform(0.3, 0.7, 1)
    u_num_lst = []
    for i, l in enumerate(l_values):
        u_num, _, _= fixed_l_inference(64, l, tol=1e-20)
        u_num_lst.append(u_num)

    plt.figure()
    plt.plot(l_values, u_num_lst, '--', linewidth=2, markersize=4)
    plt.xlabel('l')
    plt.ylabel('u_num')
    plt.title('Varying l Inference')
    plt.show()