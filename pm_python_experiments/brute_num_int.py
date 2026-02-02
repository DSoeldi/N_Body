import pm_2d_functions as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator



base_path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Disk data (for choice 2)\data'
paths = []
names = []
# additions = ["0.txt", "0_noise.txt"]
additions = ["0.txt", "1.txt", "0_noise.txt", "1_noise.txt"]

for addition in additions:
    paths.append(base_path+addition)
    names.append(addition[0:-4])

for i in range(len(paths)):
    path = paths[i]
    name = names[i]
    galaxy = pd.read_csv(path, sep = '\t', index_col=0)

    ################################
    data = name
    N = 40
    Borders = (-1,1)
    n_steps = 200
    stepsize = 0.001
    ################################

    # Generate Grid
    Grid = pm.generateGrid(N,Borders)
    Energy_list = []
    Mass_list = []
    galaxy_new = galaxy
    softening = galaxy["eps"][0]
    print(type(softening))
    pm.energyDiagnostics(galaxy_new, Grid, softening, verbose = True)
    for i in range(n_steps):
        galaxy_new = pm.leapfrog_integration_brute_force(galaxy_new, stepsize, N)
        Mass = np.sum(galaxy_new["M"])
        Energies = pm.energyDiagnostics(galaxy_new, Grid, softening, verbose = True)
        Energy_list.append(Energies)
        Mass_list.append(Mass)


    title = f"brute_galaxy_plots/galaxy{data}_N{N}_{Borders[0]}_{Borders[1]}_steps{n_steps}_stepsize{stepsize}_"

    plt.figure(figsize = [7,7])
    plt.scatter(data = galaxy_new, x="x",y="y", s = 1);
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel("x / $pc$")
    plt.ylabel("y / $pc$")
    plt.savefig(title + "galaxy_end.png", dpi = 200)
    plt.cla()

    plt.plot(Energy_list)
    plt.title("Energies")
    plt.legend(labels = ["Kinetic", "Potential", "Total"])
    plt.xlabel("Steps")
    plt.ylabel("Energy / $M_{{\odot}}\cdot pc^2 \cdot {T_{unit}}^{-2}$")
    plt.savefig(title + "Energy_plot.png", dpi = 200)
    plt.cla()

    # plt.plot(Mass_list)
    # plt.title("Masses")
    # plt.xlabel("Steps")
    # plt.ylabel("Mass / $M_{{\odot}}$")
    # plt.savefig(title + "Mass_plot.png")
    # plt.cla()
