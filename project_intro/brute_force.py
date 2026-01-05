import numpy as np
import pandas as pd
from numpy.linalg import norm

path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Data for choice 1\data.txt'
def get_galaxy_data(path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Data for choice 1\data.txt', n = 100):
    """reads the galaxy data and returns:
    the vectors m, r, v of the minigalaxy with n particles, 
    the minigalaxy as a dataframe
    the full galaxy as a dataframe"""
    df = pd.read_csv(path, sep = '\t', header = None, index_col=0)
    colnames = ["Mass", "x", "y", "z", "Vx", "Vy", "Vz", "softening", "potential"]
    df.columns = colnames
    df["radial_range"] = norm(df[["x","y","z"]].values, axis = 1)
    minigalaxy = df.sample(n = n)
    m_mini = np.array(minigalaxy.loc[:, 'Mass'])
    r_mini = np.array(minigalaxy.loc[:, 'x':'z'])
    v_mini = np.array(minigalaxy.loc[:, 'Vx':'Vz'])
    return m_mini, r_mini, v_mini, minigalaxy, df

def calc_brute_force(r, m, epsilon): 
    G = 1
    Force = np.zeros_like(r)  # three dimensional vector forces, like the radius vector
    n = len(r)
    for i in range(n):
        if i % 10 == 0: 
            print(f"Brute Progress {epsilon}: ", (i/n)*100)
        for j in range(i+1, n):
            # Vector from i to j
            d_r = r[j] - r[i]
            dist = norm(d_r)
            
            # Force magnitude, one dimension
            force_magnitude = G * m[i] * m[j] / (dist + epsilon)**2
            
            # Force vector (direction from i to j)
            force_vector = force_magnitude * (d_r / dist)
            
            Force[i] += force_vector   # positive for i
            Force[j] -= force_vector   # negative for j because of Newtons 3rd law
            
    return Force

