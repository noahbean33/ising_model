"""
Example demonstrating parallel processing with the Ising Model.
"""

import numpy as np
import matplotlib.pyplot as plt
from ising_model.parallel import temperature_sweep, analyze_phase_transition
from ising_model.utils import create_domain_wall

def main():
    print("Parallel Ising Model Simulation")
    print("==============================")
    
    # Parameters
    size = 50
    steps = 1000
    burn_in = 100
    num_temps = 10  # Number of temperature points
    
    # Create a domain wall initial state
    initial_state = create_domain_wall(size, 'horizontal')
    
    print(f"Running {num_temps} simulations in parallel...")
    
    # Run temperature sweep
    results = temperature_sweep(
        temperatures=np.linspace(1.0, 3.5, num_temps),
        size=size,
        steps=steps,
        burn_in=burn_in,
        initial_state=initial_state,
        save_results=True,
        progress=True
    )
    
    # Extract data for plotting
    temps = [r['parameters']['temperature'] for r in results]
    magnetizations = [r['observables']['mean_magnetization'] for r in results]
    energies = [r['observables']['mean_energy'] for r in results]
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(temps, magnetizations, 'o-')
    plt.axvline(x=2.269, color='r', linestyle='--', label='Critical T')
    plt.xlabel('Temperature')
    plt.ylabel('|Magnetization| per spin')
    plt.title('Magnetization vs Temperature')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(temps, energies, 'o-')
    plt.axvline(x=2.269, color='r', linestyle='--', label='Critical T')
    plt.xlabel('Temperature')
    plt.ylabel('Energy per spin')
    plt.title('Energy vs Temperature')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print execution times
    print("\nExecution times:")
    for i, r in enumerate(results):
        print(f"T = {r['parameters']['temperature']:.3f}: {r['execution_time']:.2f} seconds")

if __name__ == "__main__":
    main()
