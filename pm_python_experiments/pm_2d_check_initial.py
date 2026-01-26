import pm_2d_functions as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Disk data (for choice 2)\data1.txt'
galaxy = pd.read_csv(path, sep = '\t', index_col=0)
N = 100
print(galaxy)
# Generate Grid
Grid = pm.generateGrid(N,(-1,1))
galaxy_array = galaxy.to_records(index=False)  # Konvertieren
galaxy_x = galaxy_array['x']
galaxy_y = galaxy_array['y']
galaxy_M = galaxy_array['M']
total_mass = np.sum(galaxy_M)
# Extract and flatten grid coordinates
grid_x_flat = Grid['x'].ravel()
grid_y_flat = Grid['y'].ravel()
N = len(Grid)

Grid["mass"] = pm.populateMassGrid_CIC(galaxy_x, galaxy_y, galaxy_M, 
                                     grid_x_flat, grid_y_flat, N, total_mass)

Grid["mass"] = Grid["mass"] * total_mass

# Grid["radial_range"] = np.linalg.norm([Grid["x"], Grid["y"]])
Grid["radial_range"] = np.sqrt(Grid["x"]**2 + Grid["y"]**2)

max_rr = Grid['radial_range'].max()

radial_range_bins = np.linspace(0,max_rr,15)

# sum_integration = []
# theoretical_sum = []

# for i in range(len(radial_range_bins)):
#     print(i)
#     mask_bigger = Grid["radial_range"]<radial_range_bins[i]
#     mask_smaller = Grid["radial_range"]>radial_range_bins[i-1]
#     mask = np.bitwise_and(mask_bigger, mask_smaller)
#     masses = Grid["mass"]
#     # binsize = radial_range_bins[i]-radial_range_bins[i-1]
#     sum_integration.append(np.sum(masses[mask]))
#     theoretical_sum.append(72.5*(radial_range_bins[i]/0.01)**-2)

# plt.plot(radial_range_bins, sum_integration)
# plt.scatter(radial_range_bins, theoretical_sum)
# plt.show()

sum_integration = []
theoretical_sum = []

for i in range(1, len(radial_range_bins)):
    # Get particles in this radial bin
    mask_bigger = Grid["radial_range"] < radial_range_bins[i]
    mask_smaller = Grid["radial_range"] >= radial_range_bins[i-1]
    mask = np.bitwise_and(mask_bigger, mask_smaller)
    
    # Sum masses in bin
    total_mass_in_bin = np.sum(Grid["mass"][mask])
    sum_integration.append(total_mass_in_bin)
    
    # Calculate EXPECTED TOTAL MASS in this bin from theory
    r_center = (radial_range_bins[i] + radial_range_bins[i-1]) / 2
    sigma_at_center = 72.5 * (r_center / 0.1)**(-2)
    
    # Area of annular bin
    r_inner = radial_range_bins[i-1]
    r_outer = radial_range_bins[i]
    annular_area = np.pi * (r_outer**2 - r_inner**2)
    
    # Expected mass = surface density × area
    expected_mass = sigma_at_center * annular_area
    theoretical_sum.append(expected_mass)


plt.scatter(radial_range_bins[1:], sum_integration, label = "data radial density", color = "tab:blue")
plt.plot(radial_range_bins[1:], theoretical_sum, label = "theoretical density", color = "tab:orange")
plt.legend()
plt.xlabel("radial range / parsec")
plt.ylabel("density / solar masses / parsec^3")
plt.savefig("density_check_data_1_N100_15_bins.png")
plt.cla()