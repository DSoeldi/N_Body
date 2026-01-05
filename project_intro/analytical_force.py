import numpy as np
from numpy.linalg import norm

def calc_analytical_force(r, m, epsilon):
    n = len(r)
    G = 1
    forces = np.zeros_like(r)  # 3d vector forces like radius vector of particle
    r_all = norm(r, axis=1)
    
    for i in range(n):
        if i % 10 == 0: 
            print(f"Analytical Progress {epsilon}: ", (i/n)*100)
        r_i = r_all[i]
        # Mass enclosed, excluding particle i itself
        M_enc = np.sum(m[r_all < r_i])
        
        if r_i > 0:
            force_magnitude = G * M_enc * m[i] / (r_i+epsilon)**3 # how strong the force on particle i is
            forces[i] = -force_magnitude * (r[i] / r_i)  # which direction it points
            # negative because of attraction to the center
    return forces