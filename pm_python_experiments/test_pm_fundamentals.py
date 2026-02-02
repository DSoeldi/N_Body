import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pm_2d_functions as pm

def test_pm_fundamentals():
    """Test the basic PM calculations step by step"""
    
    # Create a simple test: 4 particles in a square
    test_particles = pd.DataFrame({
        'M': [1.0, 1.0, 1.0, 1.0],
        'x': [-0.5, 0.5, -0.5, 0.5],
        'y': [-0.5, -0.5, 0.5, 0.5],
        'z': [0.0, 0.0, 0.0, 0.0],
        'vx': [0.0, 0.0, 0.0, 0.0],
        'vy': [0.0, 0.0, 0.0, 0.0],
        'vz': [0.0, 0.0, 0.0, 0.0],
        'eps': [0.01, 0.01, 0.01, 0.01]
    })
    
    N = 10
    Borders = (-1, 1)
    Grid = pm.generateGrid(N, Borders)
    dx = 2.0 / (N-1)
    dy = 2.0 / (N-1)
    
    # Convert to array
    galaxy_array = test_particles.to_records(index=False)
    softening = galaxy_array["eps"][0]
    
    print("=== STEP 1: Mass Assignment ===")
    galaxy_x = galaxy_array['x']
    galaxy_y = galaxy_array['y']
    galaxy_M = galaxy_array['M']
    total_mass = np.sum(galaxy_M)
    
    grid_x_flat = Grid['x'].ravel()
    grid_y_flat = Grid['y'].ravel()
    
    # Manual CIC for first particle
    print(f"Particle 0 at ({galaxy_x[0]}, {galaxy_y[0]})")
    
    # Find grid cell
    x_min = grid_x_flat[0]
    x_max = grid_x_flat[-1]
    y_min = grid_y_flat[0]
    y_max = grid_y_flat[-1]
    dx_grid = (x_max - x_min) / (N - 1)
    dy_grid = (y_max - y_min) / (N - 1)
    
    i_cell = int((galaxy_x[0] - x_min) / dx_grid)
    j_cell = int((galaxy_y[0] - y_min) / dy_grid)
    print(f"  Grid cell: ({i_cell}, {j_cell})")
    print(f"  Cell center: ({x_min + i_cell*dx_grid}, {y_min + j_cell*dy_grid})")
    
    # Calculate mass grid using your function
    mass_grid = pm.populateMassGrid_CIC(galaxy_x, galaxy_y, galaxy_M, 
                                      grid_x_flat, grid_y_flat, N, total_mass)
    
    print(f"\nTotal mass in grid: {np.sum(mass_grid):.6f}")
    print(f"Total actual mass: {total_mass:.6f}")
    print(f"Mass conservation: {np.sum(mass_grid) / total_mass:.6%}")
    
    # Assign to grid
    Grid["mass"] = mass_grid
    
    print("\n=== STEP 2: Potential Calculation ===")
    # Calculate potential
    Grid = pm.calcPotential(Grid, softening, dx, dy)
    potential = Grid["potential"]
    
    print(f"Potential shape: {potential.shape}")
    print(f"Potential min/max: {np.min(potential):.6e}, {np.max(potential):.6e}")
    print(f"Potential mean: {np.mean(potential):.6e}")
    
    # Plot potential
    plt.figure(figsize=(10, 8))
    plt.imshow(potential, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    plt.colorbar(label='Potential')
    plt.scatter(galaxy_x, galaxy_y, c='red', s=50, label='Particles')
    plt.title('Potential from PM Method')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('potential_debug.png', dpi=150)
    
    print("\n=== STEP 3: Force Calculation ===")
    Fx, Fy = pm.compute_forces(Grid, dx, dy)
    
    print(f"Force Fx shape: {Fx.shape}")
    print(f"Force Fy shape: {Fy.shape}")
    print(f"Fx min/max: {np.min(Fx):.6e}, {np.max(Fx):.6e}")
    print(f"Fy min/max: {np.min(Fy):.6e}, {np.max(Fy):.6e}")
    
    # Interpolate forces to particles
    fx_pm, fy_pm = pm.interpolate_forces_to_particles(galaxy_array, Grid, Fx, Fy)
    
    print("\n=== STEP 4: Compare with Brute Force ===")
    r = np.column_stack((galaxy_x, galaxy_y))
    m = galaxy_M
    fx_bf, fy_bf = pm.calc_brute_force(r, m, softening)
    
    print("\nParticle forces comparison:")
    print(f"{'Particle':<10} {'PM Fx':<15} {'BF Fx':<15} {'Error':<15}")
    print("-" * 60)
    for i in range(len(galaxy_x)):
        error_x = abs(fx_pm[i] - fx_bf[i])
        error_y = abs(fy_pm[i] - fy_bf[i])
        print(f"{i:<10} {fx_pm[i]:<15.6e} {fx_bf[i]:<15.6e} {error_x:<15.6e}")
        print(f"{'':<10} {fy_pm[i]:<15.6e} {fy_bf[i]:<15.6e} {error_y:<15.6e}")
    
    total_error = np.sqrt(np.mean((fx_pm - fx_bf)**2 + (fy_pm - fy_bf)**2))
    print(f"\nTotal RMS force error: {total_error:.6e}")

# Run the test
test_pm_fundamentals()