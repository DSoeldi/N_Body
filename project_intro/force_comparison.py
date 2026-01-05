from brute_force import get_galaxy_data, calc_brute_force
from analytical_force import calc_analytical_force
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm
# epsilon = 0.048596998711201725
epsilon = 0.048596998711201725

path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Data for choice 1\data.txt'
m, r, v, minigalaxy, galaxy = get_galaxy_data(n = 100)
brute_force = calc_brute_force(r,m,epsilon)

analytical_force = calc_analytical_force(r,m,epsilon)
print(analytical_force.shape)
print(norm(analytical_force, axis = 1).shape)
print(minigalaxy.radial_range.shape)

plt.grid()
plt.scatter(minigalaxy.radial_range, norm(brute_force, axis = 1), label = "brute force")
plt.scatter(minigalaxy.radial_range, norm(analytical_force, axis = 1), label = "analytical force")
plt.xlabel("radial range")
plt.ylabel("force")
plt.xscale('log')
plt.legend()
plt.show()
