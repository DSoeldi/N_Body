import pm_2d_functions as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from celluloid import Camera



path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Disk data (for choice 2)\data0.txt'
galaxy = pd.read_csv(path, sep = '\t', index_col=0)
N = 300

# Generate Grid
Grid = pm.generateGrid(N,(-1,1))

fig1, ax1 = plt.subplots()
camera = Camera(fig1)
galaxy_new = galaxy
for i in range(1000):
    print(i)
    galaxy_new = pm.leapfrog_integration(galaxy_new, Grid, 0.0001, N)
    plt.scatter(data = galaxy_new, x="x",y="y", s = 1);
    camera.snap()
anim = camera.animate(blit = True)
anim.save("scatter.gif")


# fig, (ax1, ax2) = plt.subplots(1,2, figsize = [14,7])
# ax1.scatter(data = galaxy, x="x",y="y", s = 1);
# ax2.scatter(data = galaxy_new, x="x",y="y", s = 1);
# plt.show()
