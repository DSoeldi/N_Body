from brute_force import get_galaxy_data
import numpy as np

path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Data for choice 1\data.txt'
m, r, v, minigalaxy, galaxy = get_galaxy_data(n = 1000)

def calc_R_halfmass(galaxy):
    """returns halfmass radius in parsec"""
    integrated_mass = 0
    galaxy_sorted = galaxy.sort_values("radial_range")
    half_mass = sum(galaxy["Mass"])/2

    for index, row in galaxy_sorted.iterrows():
        integrated_mass+=row["Mass"]
        if integrated_mass>half_mass:
            return row["radial_range"]

halfmass_range = calc_R_halfmass(galaxy)
half_mass = sum(galaxy["Mass"])/2

print(halfmass_range)
print(half_mass)

v_c = np.sqrt(half_mass/halfmass_range) 

t_cross = halfmass_range/v_c # crossing time of galaxy 
t_relax = len(galaxy)*t_cross/(8*np.log(len(galaxy)))
print(t_relax) # relaxation time of galaxy 


