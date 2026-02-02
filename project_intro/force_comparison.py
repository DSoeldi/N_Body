from brute_force import get_galaxy_data, calc_brute_force
from analytical_force import calc_analytical_force
import matplotlib.pyplot as plt
from numpy.linalg import norm

# epsilon = 0.048596998711201725
path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Data for choice 1\data.txt'
m, r, v, minigalaxy, galaxy = get_galaxy_data(n = None)
epsilon_list = [0,0.01,0.048596998711201725,0.1]
# epsilon_list = [0.048596998711201725]

def force_plotting(radial_range, brute_force, analytical_force, epsilon):
    plt.grid()
    plt.scatter(radial_range, norm(brute_force, axis = 1), label = "brute force", s = 1)
    plt.scatter(radial_range, norm(analytical_force, axis = 1), label = "analytical force", s = 1)
    plt.xlabel("radial range / $pc$")
    plt.ylabel("Force / $M_{{\odot}}\cdot pc \cdot {T_{unit}}^{-2}$")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    # plt.ylim([0,6*1e8])
    # plt.title(f"softening = {epsilon}")
    plt.savefig(f"./project_intro/brute_force_calculations/brute_force_softening_{epsilon}.png", dpi = 200)
    plt.cla()

for epsilon in epsilon_list:
    brute_force = calc_brute_force(r,m,epsilon)
    analytical_force = calc_analytical_force(r,m,epsilon)
    force_plotting(minigalaxy.radial_range, brute_force, analytical_force, epsilon)


# the smaller the softening, the bigger the force and the less smooth the curve of the forces. 