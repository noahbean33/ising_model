"""
Simulation and visualization tools for the 2D Ising Model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple
from .model import IsingModel

def simulate_ising(
    size: int = 100,
    temperature: float = 2.269,
    steps: int = 1000,
    burn_in: int = 100,
    initial_state: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run an Ising model simulation and return the results.
    
    Args:
        size: Size of the square lattice
        temperature: Temperature in units of J/k_B
        steps: Number of Monte Carlo steps to run
        burn_in: Number of initial steps to discard
        initial_state: Optional initial state of the lattice
        
    Returns:
        Tuple of (lattice, energies, magnetizations)
    """
    model = IsingModel(size=size, temperature=temperature, initial_state=initial_state)
    energies, magnetizations = model.simulate(steps, burn_in)
    return model.lattice, energies, magnetizations

def plot_lattice(lattice: np.ndarray, title: str = "Ising Model") -> None:
    """
    Plot the current state of the lattice.
    
    Args:
        lattice: 2D array of spins (-1 or 1)
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(lattice, cmap='binary')
    plt.title(title)
    plt.colorbar(label='Spin')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_observables(energies: np.ndarray, magnetizations: np.ndarray, temperature: float) -> None:
    """
    Plot energy and magnetization over time.
    
    Args:
        energies: Array of energy values
        magnetizations: Array of magnetization values
        temperature: Temperature of the simulation
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Energy plot
    ax1.plot(energies / len(energies), 'b-')
    ax1.set_ylabel('Energy per spin')
    ax1.set_title(f'Ising Model at T = {temperature:.3f}')
    
    # Magnetization plot
    ax2.plot(np.abs(magnetizations) / magnetizations.size, 'r-')
    ax2.set_xlabel('Monte Carlo Steps')
    ax2.set_ylabel('|Magnetization| per spin')
    
    plt.tight_layout()
    plt.show()

class IsingAnimation:
    """Class for creating animations of the Ising model simulation."""
    
    def __init__(self, model: IsingModel, steps: int = 1000, interval: int = 50):
        """
        Initialize the animation.
        
        Args:
            model: Initialized IsingModel instance
            steps: Number of steps to animate
            interval: Delay between frames in milliseconds
        """
        self.model = model
        self.steps = steps
        self.interval = interval
        
        # Set up the figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.suptitle(f'Ising Model at T = {model.temperature:.3f}')
        
        # Lattice plot
        self.img = self.ax1.imshow(model.lattice, cmap='binary')
        self.ax1.set_title('Lattice')
        self.ax1.axis('off')
        
        # Energy and magnetization plot
        self.energy_line, = self.ax2.plot([], [], 'b-', label='Energy')
        self.mag_line, = self.ax2.plot([], [], 'r-', label='|Magnetization|')
        self.ax2.set_xlim(0, steps)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_xlabel('Monte Carlo Steps')
        self.ax2.legend()
        
        # Data storage
        self.energies = []
        self.magnetizations = []
        
    def _init_animation(self):
        """Initialize the animation."""
        self.energies = []
        self.magnetizations = []
        self.energy_line.set_data([], [])
        self.mag_line.set_data([], [])
        return self.img, self.energy_line, self.mag_line
    
    def _update(self, frame):
        """Update the animation for each frame."""
        # Perform one Monte Carlo step
        self.model.step()
        
        # Update lattice image
        self.img.set_array(self.model.lattice)
        
        # Store and update energy and magnetization
        self.energies.append(self.model.energy / self.model.lattice.size)
        self.magnetizations.append(abs(self.model.magnetization) / self.model.lattice.size)
        
        # Update plots
        x = range(len(self.energies))
        self.energy_line.set_data(x, self.energies)
        self.mag_line.set_data(x, self.magnetizations)
        
        # Adjust y-axis limits
        if len(self.energies) > 1:
            min_e, max_e = min(self.energies), max(self.energies)
            min_m, max_m = min(self.magnetizations), max(self.magnetizations)
            self.ax2.set_ylim(
                min(min_e, min_m) * 0.95,
                max(max_e, max_m) * 1.05
            )
        
        return self.img, self.energy_line, self.mag_line
    
    def animate(self, save_path: Optional[str] = None):
        """
        Run the animation.
        
        Args:
            save_path: If provided, save the animation to this path
        """
        anim = FuncAnimation(
            self.fig, 
            self._update,
            init_func=self._init_animation,
            frames=self.steps,
            interval=self.interval,
            blit=True,
            repeat=False
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=30)
        
        plt.tight_layout()
        plt.show()
        return anim
