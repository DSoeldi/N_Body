import numpy as np
from scipy.fft import fft2, ifft2
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
import pandas as pd
import numba

def generateGrid(N:int,Boundaries:tuple):
    """
    Generate a linearly spaced 2D grid with N gridpoints per dimension from the boundary left to the boundary right
    """
    left, right = Boundaries
    Coordinates = np.round(np.linspace(left,right,N),int(np.log10(N)+1))
    X, Y = np.meshgrid(Coordinates, Coordinates)
    # Generate the grid with predetermined types
    Grid = np.zeros((N, N), dtype=[('x', float), ('y', float), ('mass', float), ('potential', float), ('radial_range', float)])
    # Populate the grid with the coordinate values
    Grid['x'], Grid['y'] = X, Y
    return Grid

# def oneNodeMassAssignment(Particle, Grid, total_mass):
#     """
#     xxx
#     """
#     N = len(Grid)
#     particle_position = np.array(Particle[["x", "y"]])
#     gridpoints = np.column_stack((Grid["x"].ravel(), Grid["y"].ravel())) 
#     distances = np.linalg.norm(gridpoints - particle_position, axis = 1)
#     closest_coord = np.argmin(distances) # index of minimal distance in the stacked coordinate array
#     closest_index = np.unravel_index(closest_coord, (N,N)) # return indexes in NxN array
#     Grid["mass"][closest_index] += Particle["M"]/total_mass
#     return Grid

# def oneNodeMassAssignment(galaxy_array, Grid, total_mass):
#     """
#     xxx
#     """
#     N = len(Grid)
#     gridpoints = np.column_stack((Grid["x"].ravel(), Grid["y"].ravel())) 
#     tree = cKDTree(gridpoints)
#     particle_points = np.column_stack((galaxy_array['x'], galaxy_array['y']))
#     _, indices = tree.query(particle_points)
#     print(indices)
#     return Grid

@numba.jit(nopython=True)
def populateMassGrid_JIT(galaxy_x, galaxy_y, galaxy_M, grid_x_flat, grid_y_flat, N, total_mass):
    mass_grid = np.zeros((N, N))
    
    for i in range(len(galaxy_x)):
        min_dist = np.inf
        min_idx = 0
        
        # Find closest grid point
        for j in range(len(grid_x_flat)):
            dx = grid_x_flat[j] - galaxy_x[i]
            dy = grid_y_flat[j] - galaxy_y[i]
            dist = dx*dx + dy*dy
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        
        row = min_idx // N
        col = min_idx % N
        mass_grid[row, col] += galaxy_M[i] / total_mass
    
    return mass_grid

# version without dividing through area
@numba.jit(nopython=True)
def populateMassGrid_CIC(galaxy_x, galaxy_y, galaxy_M, grid_x_flat, grid_y_flat, N, total_mass):
    """
    Cloud-in-Cell mass assignment: distribute mass to 4 nearest grid points
    using bilinear interpolation weights.
    """
    mass_grid = np.zeros((N, N))
    
    # Extract grid bounds and spacing from flattened coordinates
    # Assuming regular grid
    x_min = grid_x_flat[0]
    x_max = grid_x_flat[-1]
    y_min = grid_y_flat[0]
    y_max = grid_y_flat[-1]
    
    dx = (x_max - x_min) / (N - 1)
    dy = (y_max - y_min) / (N - 1)
    
    for i in range(len(galaxy_x)):
        px = galaxy_x[i]
        py = galaxy_y[i]
        mass = galaxy_M[i] / total_mass
        
        # Find the grid cell (lower-left corner indices)
        # This gives us the index of the grid point to the lower-left of the particle
        i_cell = int((px - x_min) / dx)
        j_cell = int((py - y_min) / dy)
        
        # Clamp to valid grid range (avoid boundary issues)
        if i_cell < 0:
            i_cell = 0
        elif i_cell >= N - 1:
            i_cell = N - 2
            
        if j_cell < 0:
            j_cell = 0
        elif j_cell >= N - 1:
            j_cell = N - 2
        
        # Calculate fractional position within the cell (0 to 1)
        x_cell = x_min + i_cell * dx
        y_cell = y_min + j_cell * dy
        
        fx = (px - x_cell) / dx
        fy = (py - y_cell) / dy
        
        # Clamp fractional positions to [0, 1]
        if fx < 0.0:
            fx = 0.0
        elif fx > 1.0:
            fx = 1.0
            
        if fy < 0.0:
            fy = 0.0
        elif fy > 1.0:
            fy = 1.0
        
        # Distribute mass to 4 corners using bilinear weights
        # (i_cell, j_cell) is bottom-left in grid indices
        mass_grid[j_cell, i_cell] += mass * (1.0 - fx) * (1.0 - fy)
        mass_grid[j_cell, i_cell + 1] += mass * fx * (1.0 - fy)
        mass_grid[j_cell + 1, i_cell] += mass * (1.0 - fx) * fy
        mass_grid[j_cell + 1, i_cell + 1] += mass * fx * fy
    
    return mass_grid

# # version with dividing through area
# @numba.jit(nopython=True)
# def populateMassGrid_CIC(galaxy_x, galaxy_y, galaxy_M, grid_x_flat, grid_y_flat, N, total_mass):
#     """
#     Cloud-in-Cell mass assignment: distribute mass to 4 nearest grid points
#     using bilinear interpolation weights and calculate the actual density.
#     """
#     mass_grid = np.zeros((N, N))
    
#     # Extract grid bounds and spacing from flattened coordinates
#     # Assuming regular grid
#     x_min = grid_x_flat[0]
#     x_max = grid_x_flat[-1]
#     y_min = grid_y_flat[0]
#     y_max = grid_y_flat[-1]
    
#     dx = (x_max - x_min) / (N - 1)
#     dy = (y_max - y_min) / (N - 1)
#     cell_area = dx * dy  # Area of each grid cell
    
#     for i in range(len(galaxy_x)):
#         px = galaxy_x[i]
#         py = galaxy_y[i]
#         mass = galaxy_M[i] / total_mass
        
#         # Find the grid cell (lower-left corner indices)
#         # This gives us the index of the grid point to the lower-left of the particle
#         i_cell = int((px - x_min) / dx)
#         j_cell = int((py - y_min) / dy)
        
#         # Clamp to valid grid range (avoid boundary issues)
#         if i_cell < 0:
#             i_cell = 0
#         elif i_cell >= N - 1:
#             i_cell = N - 2
            
#         if j_cell < 0:
#             j_cell = 0
#         elif j_cell >= N - 1:
#             j_cell = N - 2
        
#         # Calculate fractional position within the cell (0 to 1)
#         x_cell = x_min + i_cell * dx
#         y_cell = y_min + j_cell * dy
        
#         fx = (px - x_cell) / dx
#         fy = (py - y_cell) / dy
        
#         # Clamp fractional positions to [0, 1]
#         if fx < 0.0:
#             fx = 0.0
#         elif fx > 1.0:
#             fx = 1.0
            
#         if fy < 0.0:
#             fy = 0.0
#         elif fy > 1.0:
#             fy = 1.0
        
#         # Distribute mass to 4 corners using bilinear weights
#         # (i_cell, j_cell) is bottom-left in grid indices
#         mass_grid[j_cell, i_cell] += mass * (1.0 - fx) * (1.0 - fy)
#         mass_grid[j_cell, i_cell + 1] += mass * fx * (1.0 - fy)
#         mass_grid[j_cell + 1, i_cell] += mass * (1.0 - fx) * fy
#         mass_grid[j_cell + 1, i_cell + 1] += mass * fx * fy
    
#     # Calculate the actual density by dividing the mass by the cell area
#     density_grid = mass_grid / cell_area
    
#     return density_grid


# @numba.jit(nopython = True)
# def populateMassGrid_JIT(galaxy_array, Grid, total_mass):
#     """
#     xxx
#     """
#     length = len(galaxy_array)
#     N = len(Grid)
#     Grid["mass"].fill(0)
#     for particle in galaxy_array:
#         particle_position = np.array([particle['x'], particle['y']])
#         gridpoints = np.column_stack((Grid['x'].ravel(), Grid['y'].ravel()))

#         # Compute distances manually
#         distances = np.zeros(len(gridpoints))
#         for i in range(len(gridpoints)):
#             dx = gridpoints[i, 0] - particle_position[0]
#             dy = gridpoints[i, 1] - particle_position[1]
#             distances[i] = np.sqrt(dx*dx + dy*dy)

#         closest_coord = np.argmin(distances)
#         # Manual conversion from flat index to 2D index
#         row = closest_coord // N
#         col = closest_coord % N
#         Grid['mass'][row, col] += particle['M'] / total_mass

#     return Grid

def calcPotential(Grid):
    """
    Solve the poisson equation using fast fourier transform to generate the potential field. 
    """
    mass_density = Grid["mass"]

    density_fourier_space = fft2(mass_density) # change mass density into fourier space
    # Compute the wave numbers
    nx, ny = mass_density.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny)
    Kx, Ky = np.meshgrid(kx, ky)
    k_squared = Kx**2 + Ky**2 +(2 * np.pi / 0.118)**2 ### softening squared?

    # Avoid division by zero at k=0
    k_squared[0, 0] = 1.0

    # Compute the potential in frequency space
    potential_fft = 2 * np.pi * 1 * density_fourier_space / k_squared # G = 1

    # # Apply softening by modifying the kernel
    # softening_kernel = np.exp(-0.118 * np.sqrt(k_squared))
    # potential_fft *= softening_kernel

    # Perform the inverse FFT
    Grid["potential"] = np.real(ifft2(potential_fft))

    return Grid

def compute_forces(Grid, dx, dy):
    """
    Compute the gravitational forces from the potential using central differences.
    
    Parameters:
    - Grid: dictionary or structured array with 'potential' field
    - dx (float): Grid spacing in the x direction
    - dy (float): Grid spacing in the y direction
    
    Returns:
    - Fx (np.ndarray): 2D array representing the x-component of the force
    - Fy (np.ndarray): 2D array representing the y-component of the force
    """
    # Reshape potential if needed
    potential = Grid['potential']
    
    # Compute the gradient of the potential (force = -grad(potential))
    # axis=1 is x-direction, axis=0 is y-direction
    Fx = -np.gradient(potential, dx, axis=1)
    Fy = -np.gradient(potential, dy, axis=0)

    # # Apply the softening factor to the distance
    # softened_distance = np.sqrt(dx**2 + dy**2)

    # # Normalize the force by the softened distance
    # Fx /= softened_distance
    # Fy /= softened_distance
    
    return Fx, Fy

def fullGridCalc(galaxy_array, Grid, dx, dy):
    """returns the forces in x and y directions from the galaxy and the grid
    this calculation is needed before the acceleration step in the leap frog scheme"""
    galaxy_x = galaxy_array['x']
    galaxy_y = galaxy_array['y']
    galaxy_M = galaxy_array['M']
    total_mass = np.sum(galaxy_M)
    
    # Extract and flatten grid coordinates
    grid_x_flat = Grid['x'].ravel()
    grid_y_flat = Grid['y'].ravel()
    N = len(Grid)
    Grid["mass"] = populateMassGrid_CIC(galaxy_x, galaxy_y, galaxy_M, 
                                     grid_x_flat, grid_y_flat, N, total_mass)    
    Grid = calcPotential(Grid)
    return compute_forces(Grid, dx, dy) # (Fx, Fy)

def interpolate_forces_to_particles(particles, Grid, Fx, Fy):
    """
    Interpolate grid-based forces to particle positions using bilinear interpolation.
    
    Parameters:
    - particles: structured array with 'x' and 'y' fields
    - Grid: structured array with grid coordinates
    - Fx, Fy: 2D arrays of forces on the grid
    
    Returns:
    - fx, fy: 1D arrays of forces at particle positions
    """
    
    # Extract unique grid coordinates
    x_grid = np.unique(Grid['x'])
    y_grid = np.unique(Grid['y'])
    
    # Create interpolators for the force components
    interp_fx = RegularGridInterpolator((y_grid, x_grid), Fx, method='linear', bounds_error=False, fill_value=0)
    interp_fy = RegularGridInterpolator((y_grid, x_grid), Fy, method='linear', bounds_error=False, fill_value=0)
    
    # Evaluate forces at particle positions
    particle_coords = np.column_stack((particles['y'], particles['x']))
    fx = interp_fx(particle_coords)
    fy = interp_fy(particle_coords)
    
    return fx, fy

def leapfrog_integration(galaxy, Grid, dt, N=100):
    """
    Perform a single leapfrog timestep for all particles simultaneously.
    
    The leapfrog algorithm:
    1. v(t + dt/2) = v(t) + a(t) * dt/2  (kick)
    2. x(t + dt) = x(t) + v(t + dt/2) * dt  (drift)
    3. v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2  (kick)
    
    For the first step, we use a simplified version (kick-drift-kick).
    
    Parameters:
    - galaxy_array: structured array with particle data ('x', 'y', 'vx', 'vy', etc.)
    - Grid: structured array with grid data including 'potential'
    - dt (float): Timestep size
    - N (int): Grid resolution (default 100)
    
    Returns:
    - galaxy_array: Updated particle array with new positions and velocities
    """
    dx = 2.0 / N
    dy = 2.0 / N
    galaxy_array = galaxy.to_records(index=False)  # Konvertieren

    # Compute forces on the grid
    Fx, Fy = fullGridCalc(galaxy_array, Grid, dx, dy)

    # Interpolate forces to particle positions
    fx, fy = interpolate_forces_to_particles(galaxy_array, Grid, Fx, Fy)
    
    # Convert forces to accelerations (assuming unit mass or mass already included in force)
    ax = fx / galaxy_array["M"]
    ay = fy / galaxy_array["M"]
    
    # Leapfrog step: kick-drift-kick
    # Half-step velocity update (kick)
    galaxy_array['vx'] += 0.5 * ax * dt
    galaxy_array['vy'] += 0.5 * ay * dt
    
    # Full-step position update (drift)
    galaxy_array['x'] += galaxy_array['vx'] * dt
    galaxy_array['y'] += galaxy_array['vy'] * dt
    
    # Recompute forces at new positions
    # Compute forces on the grid
    # galaxy = pd.DataFrame(galaxy_array, columns=['M', 'x', 'y', 'z', 'vx','vy', 'vz', 'eps'])
    Fx, Fy = fullGridCalc(galaxy_array, Grid, dx, dy)
    # galaxy_array = galaxy.to_records(index=False)  # Konvertieren

    # Interpolate forces to particle positions
    fx_new, fy_new = interpolate_forces_to_particles(galaxy_array, Grid, Fx, Fy)
    
    # Convert forces to accelerations (assuming unit mass or mass already included in force)
    ax_new = fx_new / galaxy_array["M"]
    ay_new = fy_new / galaxy_array["M"]

    # Half-step velocity update (kick)
    galaxy_array['vx'] += 0.5 * ax_new * dt
    galaxy_array['vy'] += 0.5 * ay_new * dt
    galaxy = pd.DataFrame(galaxy_array, columns=['M', 'x', 'y', 'z', 'vx','vy', 'vz', 'eps'])
    return galaxy


# def populateMassGrid(galaxy_array, Grid):
#     """
#     xxx
#     """
#     length = len(galaxy_array)
#     total_mass = galaxy_array.M.sum()
#     Grid["mass"] = np.zeros_like(Grid["mass"])
#     for particle in galaxy_array.iterrows():
#         # if index % 10 == 0:
#         #     print(f"progress: {round(100*index/length, 1)}% done")
#         Grid = oneNodeMassAssignment(particle, Grid, total_mass)
#     return Grid


###################### diagnostics

def diagnosticsKineticEnergy(galaxy):
    mass = galaxy.M
    vx = galaxy.vx
    vy = galaxy.vy
    return 0.5 * np.sum(mass * (vx**2+vy**2))

def diagnosticsPotentialEnergy(Grid):
    PE = 0.5 * np.sum(Grid['mass'] * Grid['potential']) * np.sum(Grid['mass'])
    return PE

import sys
from numpy.linalg import norm
def energyDiagnostics(galaxy, Grid, verbose = False):
    KE = diagnosticsKineticEnergy(galaxy)
    r = np.array(galaxy.loc[:, 'x':'y'])
    m = np.array(galaxy.loc[:, 'M'])
    PE = np.sum(calc_potential_energy_brute(r, m, 0.118))
    TE = KE+PE
    if verbose:
        print(f"KE: {KE}, PE: {PE}, TE: {TE}")
    return KE, PE, TE



#####################################################
@numba.jit(nopython=True)
def calc_potential_energy_brute(r, m, epsilon):
    """
    Calculate gravitational potential energy using direct summation.
    
    PE = -G * Σ(i<j) [m_i * m_j / sqrt(r_ij² + ε²)]
    
    Parameters:
    - r: particle positions (N x 2 array)
    - m: particle masses (N array)
    - epsilon: softening length
    
    Returns:
    - Total potential energy (should be negative)
    """
    G = 1.0
    PE = 0.0
    n = len(r)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Vector separation
            dx = r[j, 0] - r[i, 0]
            dy = r[j, 1] - r[i, 1]
            
            # Distance with softening
            dist_squared = dx**2 + dy**2 + epsilon**2
            dist = np.sqrt(dist_squared)
            
            # Potential energy contribution (negative!)
            PE -= G * m[i] * m[j] / dist
    
    return PE