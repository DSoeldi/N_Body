import pm_2d_functions as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator



base_path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Disk data (for choice 2)\data'
paths = []
names = []
additions = ["0.txt"]
# additions = ["0.txt", "1.txt", "0_noise.txt", "1_noise.txt"]

for addition in additions:
    paths.append(base_path+addition)
    names.append(addition[0:-4])

for i in range(len(paths)):
    path = paths[i]
    name = names[i]
    galaxy = pd.read_csv(path, sep = '\t', index_col=0)

    ################################
    velocity_factor = 1
    data = name
    N = 30
    Borders = (-1,1)
    n_steps = 200
    stepsize = 0.001
    ################################
    galaxy.vx = galaxy.vx*velocity_factor
    galaxy.vy = galaxy.vy*velocity_factor

    # Generate Grid
    Grid = pm.generateGrid(N,Borders)
    Energy_list = []
    Mass_list = []
    Velocity_list = []
    galaxy_new = galaxy
    softening = galaxy["eps"][0]
    pm.energyDiagnostics(galaxy_new, Grid,softening, verbose = False)
    for i in range(n_steps):
        print(round(100*i/n_steps,0))
        galaxy_new = pm.leapfrog_integration(galaxy_new, Grid, stepsize, N)
        Mass = np.sum(galaxy_new["M"])
        Velocity = np.linalg.norm([galaxy_new["vx"], galaxy_new["vy"]])
        Energies = pm.energyDiagnostics(galaxy_new, Grid, softening, verbose = False)
        Energy_list.append(Energies)
        Mass_list.append(Mass)
        Velocity_list.append(Velocity)


    title = f"galaxy_plots/g{data}_N{N}_{Borders[0]}_{Borders[1]}_stps{n_steps}_dt{stepsize}__vf{velocity_factor}"

    plt.figure(figsize = [7,7])
    plt.scatter(data = galaxy_new, x="x",y="y", s = 1);
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    # plt.show()
    plt.savefig(title + "galaxy_end.png", dpi = 200)
    plt.cla()

    plt.plot(Energy_list)
    plt.title("Energies")
    plt.legend(labels = ["Kinetic", "Potential", "Total"])
    plt.xlabel("Steps")
    plt.ylabel("Energy / $M_{{\odot}}\cdot pc^2 \cdot {T_{unit}}^{-2}$")
    # plt.show()
    plt.savefig(title + "Energy_plot.png", dpi = 200)
    plt.cla()

    plt.plot(Mass_list)
    plt.title("Masses")
    plt.xlabel("Steps")
    plt.ylabel("Mass / $M_{{\odot}}$")
    plt.savefig(title + "Mass_plot.png")
    plt.cla()

    plt.plot(Velocity_list)
    # plt.title("Total Velocity")
    # plt.legend(labels = ["Kinetic", "Potential", "Total"])
    plt.xlabel("Steps")
    plt.ylabel("Total Velocity / $v_{unit}$")
    plt.savefig(f"galaxy_plots/velocity_plots/g{data}_N{N}_{Borders[0]}_{Borders[1]}_stps{n_steps}_dt{stepsize}__vf{velocity_factor}" + "Velocity_plot.png", dpi = 200)
    # plt.show()
    plt.cla()


virial_ratio = 0.7625