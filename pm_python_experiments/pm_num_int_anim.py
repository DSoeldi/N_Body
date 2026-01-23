import pm_2d_functions as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from celluloid import Camera



path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Disk data (for choice 2)\data0.txt'
galaxy = pd.read_csv(path, sep = '\t', index_col=0)

################################
data = "0"
N = 400
Borders = (-2,2)
n_steps = 400
stepsize = 0.001
################################

# Generate Grid
Grid = pm.generateGrid(N,Borders)

fig1, ax1 = plt.subplots()
camera = Camera(fig1)
galaxy_new = galaxy
for i in range(n_steps):
    print(i)
    galaxy_new = pm.leapfrog_integration(galaxy_new, Grid, stepsize, N)
    plt.scatter(data = galaxy_new, x="x",y="y", s = 1, color = 'k');
    camera.snap()
anim = camera.animate(blit = True, interval = 1)
anim.save(f"galaxy_animations/galaxy{data}_N{N}_{Borders[0]}_{Borders[1]}_steps{n_steps}_stepsize{stepsize}.gif", fps = 200)


# fig, (ax1, ax2) = plt.subplots(1,2, figsize = [14,7])
# ax1.scatter(data = galaxy, x="x",y="y", s = 1);
# ax2.scatter(data = galaxy_new, x="x",y="y", s = 1);
# plt.show()
