import pm_2d_functions as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator



path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Disk data (for choice 2)\data0.txt'
galaxy = pd.read_csv(path, sep = '\t', index_col=0)

################################
data = "0"
N = 200
Borders = (-1,1)
n_steps = 500
stepsize = 0.001
################################

# Generate Grid
Grid = pm.generateGrid(N,Borders)
Energy_list = []
Mass_list = []
galaxy_new = galaxy
pm.energyDiagnostics(galaxy_new, Grid, verbose = True)
for i in range(n_steps):
    galaxy_new = pm.leapfrog_integration(galaxy_new, Grid, stepsize, N)
    Mass = np.sum(galaxy_new["M"])
    Energies = pm.energyDiagnostics(galaxy_new, Grid, verbose = True)
    Energy_list.append(Energies)
    Mass_list.append(Mass)


title = f"galaxy_plots/galaxy{data}_N{N}_{Borders[0]}_{Borders[1]}_steps{n_steps}_stepsize{stepsize}_"

plt.scatter(data = galaxy_new, x="x",y="y", s = 1);
plt.savefig(title + "galaxy_end.png")
plt.cla()
plt.plot(Energy_list)
plt.title("Energies")
plt.legend(labels = ["Kinetic", "Potential", "Total"])
plt.xlabel("steps")
plt.ylabel("energy")
plt.savefig(title + "Energy_plot.png")
plt.cla()
# plt.plot(Mass_list)
# plt.title("Masses")
# plt.xlabel("steps")
# plt.ylabel("Mass")
# plt.savefig(title + "Mass_plot.png")
# plt.cla()
