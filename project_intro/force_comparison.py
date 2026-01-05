from brute_force import get_galaxy_data, calc_brute_force
from analytical_force import calc_analytical_force
import matplotlib.pyplot as plt
from numpy.linalg import norm
# epsilon = 0.048596998711201725
path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Data for choice 1\data.txt'
m, r, v, minigalaxy, galaxy = get_galaxy_data(n = 1000)

epsilon_list = [0,0.01,0.02,0.03,0.04,0.048596998711201725,0.05,0.06,0.07,0.08,0.09,0.1]

def force_plotting(radial_range, brute_force, analytical_force, epsilon):
    plt.grid()
    plt.scatter(radial_range, norm(brute_force, axis = 1), label = "brute force")
    plt.scatter(radial_range, norm(analytical_force, axis = 1), label = "analytical force")
    plt.xlabel("radial range")
    plt.ylabel("force")
    plt.xscale('log')
    plt.legend()
    plt.title(f"softening = {epsilon}")
    plt.savefig(f"./project_intro/brute_force_calculations/brute_force_softening_{epsilon}.png")
    plt.cla()

for epsilon in epsilon_list:
    brute_force = calc_brute_force(r,m,epsilon)
    analytical_force = calc_analytical_force(r,m,epsilon)
    force_plotting(minigalaxy.radial_range, brute_force, analytical_force, epsilon)




    
