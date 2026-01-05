import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm

path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Data for choice 1\data.txt'
def get_galaxy_data(path, n = 100):
    """reads the galaxy data and returns:
    the vectors m, r, v of the minigalaxy with n particles, 
    the minigalaxy as a dataframe
    the full galaxy as a dataframe"""
    df = pd.read_csv(path, sep = '\t', header = None, index_col=0)
    colnames = ["Mass", "x", "y", "z", "Vx", "Vy", "Vz", "softening", "potential"]
    df.columns = colnames
    minigalaxy = df.sample(n = n)
    m_mini = np.array(minigalaxy.loc[:, 'Mass'])
    r_mini = np.array(minigalaxy.loc[:, 'x':'z'])
    v_mini = np.array(minigalaxy.loc[:, 'Vx':'Vz'])
    return m_mini, r_mini, v_mini, minigalaxy, df

def calc_force(r,m, epsilon): 
    G = 1
    Force = np.zeros_like(m)
    n = len(r)
    for i in range(n):
        print("Progress: ", (i/n)*100)
        for j in range(i+1, n):
            # distances between two particles
            d_rx = r[j][0] - r[i][0]
            d_ry = r[j][1] - r[i][1]
            d_rz = r[j][2] - r[i][2]
            r2 = 1/np.power(norm([d_rx, d_ry, d_rz]) + epsilon, 2)
            force = m[i]*m[j]*r2
            Force[i] += force
            Force[j] -= force
    return Force

# m, r, v, minigalaxy, galaxy = get_galaxy_data(path, n = 10)
# epsilon = 0.048596998711201725
# Force = calc_force(r, m, epsilon)

# print(Force)





# plt.scatter(data = minigalaxy, x="x",y="y", s = 1);
# plt.show()


