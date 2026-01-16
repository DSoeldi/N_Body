import numpy as np
from scipy.fft import fft2, ifft2
from scipy.interpolate import RegularGridInterpolator
import pandas as pd

def generateGrid(N:int,Boundaries:tuple):
    """
    Generate a linearly spaced 2D grid with N gridpoints per dimension from the boundary left to the boundary right
    """
    left, right = Boundaries
    Coordinates = np.round(np.linspace(left,right,N),int(np.log10(N)+1))
    X, Y = np.meshgrid(Coordinates, Coordinates)
    # Generate the grid with predetermined types
    Grid = np.zeros((N, N), dtype=[('x', float), ('y', float), ('mass', float), ('potential', float)])
    # Populate the grid with the coordinate values
    Grid['x'], Grid['y'] = X, Y
    return Grid

def oneNodeMassAssignment(Particle, Grid, total_mass):
    """
    xxx
    """
    N = len(Grid)
    particle_position = np.array(Particle[["x", "y"]])
    gridpoints = np.column_stack((Grid["x"].ravel(), Grid["y"].ravel())) 
    distances = np.linalg.norm(gridpoints - particle_position, axis = 1)
    closest_coord = np.argmin(distances) # index of minimal distance in the stacked coordinate array
    closest_index = np.unravel_index(closest_coord, (N,N)) # return indexes in NxN array
    Grid["mass"][closest_index] += Particle["M"]/total_mass
    return Grid

def populateMassGrid(galaxy, Grid):
    """
    xxx
    """
    length = len(galaxy)
    total_mass = galaxy.M.sum()
    Grid["mass"] = np.zeros_like(Grid["mass"])
    for index, particle in galaxy.iterrows():
        # if index % 10 == 0:
        #     print(f"progress: {round(100*index/length, 1)}% done")
        Grid = oneNodeMassAssignment(particle, Grid, total_mass)
    return Grid

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
    k_squared = Kx**2 + Ky**2

    # Avoid division by zero at k=0
    k_squared[0, 0] = 1.0

    # Compute the potential in frequency space
    potential_fft = 4 * np.pi * 1 * density_fourier_space / k_squared # G = 1

    # Perform the inverse FFT
    Grid["potential"] = np.real(ifft2(potential_fft))

    return Grid


########################## Claude


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
    
    return Fx, Fy

def fullGridCalc(galaxy, Grid, dx, dy):
    """returns the forces in x and y directions from the galaxy and the grid
    this calculation is needed before the acceleration step in the leap frog scheme"""
    Grid = populateMassGrid(galaxy, Grid)
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

    # Compute forces on the grid
    Fx, Fy = fullGridCalc(galaxy, Grid, dx, dy)
    galaxy_array = galaxy.to_records(index=False)  # Konvertieren

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
    galaxy = pd.DataFrame(galaxy_array, columns=['M', 'x', 'y', 'z', 'vx','vy', 'vz', 'eps'])
    Fx, Fy = fullGridCalc(galaxy, Grid, dx, dy)
    galaxy_array = galaxy.to_records(index=False)  # Konvertieren

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


