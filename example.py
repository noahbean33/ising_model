"""
Example usage of the Ising Model package.
"""

import numpy as np
import matplotlib.pyplot as plt
from ising_model.model import IsingModel
from ising_model.simulation import IsingAnimation, plot_observables
from ising_model.utils import create_domain_wall, calculate_observables

def run_simulation():
    """Run a basic Ising model simulation with visualization."""
    # Parameters
    size = 50
    temperature = 2.0  # Below critical temperature (T_c ≈ 2.269)
    
    # Create model with random initial state
    print("Creating Ising model...")
    model = IsingModel(size=size, temperature=temperature)
    
    # Create and run animation
    print("Starting simulation... (this may take a moment)")
    animation = IsingAnimation(model, steps=100, interval=50)
    animation.animate()
    
    # Run a longer simulation and plot observables
    print("Running longer simulation...")
    energies, magnetizations = model.simulate(steps=1000, burn_in=100)
    
    # Calculate and print some observables
    obs = calculate_observables(energies, magnetizations, temperature, equilibration_time=100)
    print("\nSimulation results:")
    for key, value in obs.items():
        print(f"{key}: {value:.4f}")
    
    # Plot observables over time
    plot_observables(energies, magnetizations, temperature)
    plt.show()

def run_domain_wall_example():
    """Example showing domain wall dynamics."""
    size = 50
    temperature = 1.5  # Low temperature to see domain wall dynamics
    
    # Create a domain wall initial state
    initial_state = create_domain_wall(size, 'vertical')
    
    # Create and run the model
    model = IsingModel(size=size, temperature=temperature, initial_state=initial_state)
    
    # Create animation
    print("Simulating domain wall dynamics...")
    animation = IsingAnimation(model, steps=200, interval=50)
    animation.animate()

if __name__ == "__main__":
    print("Ising Model Simulation")
    print("======================")
    
    while True:
        print("\nChoose an example to run:")
        print("1. Basic simulation with random initial state")
        print("2. Domain wall dynamics")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            run_simulation()
        elif choice == '2':
            run_domain_wall_example()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
