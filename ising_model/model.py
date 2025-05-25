"""
Core implementation of the 2D Ising Model using the Metropolis algorithm.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

class IsingModel:
    """
    2D Ising Model implementation using the Metropolis algorithm.
    
    Attributes:
        size (int): Size of the square lattice (size x size)
        temperature (float): Temperature of the system in units of J/k_B
        lattice (np.ndarray): 2D array representing the spin configuration
        energy (float): Current energy of the system
        magnetization (float): Current magnetization of the system
        _neighbors (np.ndarray): Precomputed neighbor indices for fast lookup
    """
    
    def __init__(self, size: int = 100, temperature: float = 2.269, initial_state: Optional[np.ndarray] = None):
        """
        Initialize the Ising model.
        
        Args:
            size: Size of the square lattice (size x size)
            temperature: Temperature of the system in units of J/k_B
            initial_state: Optional initial state of the lattice. If None, a random configuration is generated.
        """
        self.size = size
        self.temperature = temperature
        
        # Initialize lattice
        if initial_state is not None:
            if initial_state.shape != (size, size):
                raise ValueError(f"Initial state must be of shape ({size}, {size})")
            self.lattice = initial_state
        else:
            self.lattice = 2 * np.random.randint(2, size=(size, size)) - 1  # Random spins: -1 or 1
        
        # Precompute neighbor indices for faster access
        self._precompute_neighbors()
        
        # Calculate initial energy and magnetization
        self.energy = self._calculate_energy()
        self.magnetization = np.sum(self.lattice)
    
    def _precompute_neighbors(self) -> None:
        """Precompute neighbor indices for periodic boundary conditions."""
        size = self.size
        self._neighbors = np.zeros((size, size, 4, 2), dtype=np.int32)
        
        for i in range(size):
            for j in range(size):
                # Right neighbor (i+1, j)
                self._neighbors[i, j, 0] = [(i + 1) % size, j]
                # Left neighbor (i-1, j)
                self._neighbors[i, j, 1] = [(i - 1) % size, j]
                # Bottom neighbor (i, j+1)
                self._neighbors[i, j, 2] = [i, (j + 1) % size]
                # Top neighbor (i, j-1)
                self._neighbors[i, j, 3] = [i, (j - 1) % size]
    
    def _calculate_energy(self) -> float:
        """
        Calculate the total energy of the current configuration.
        
        Returns:
            float: Total energy of the system
        """
        energy = 0.0
        for i in range(self.size):
            for j in range(self.size):
                spin = self.lattice[i, j]
                # Get precomputed neighbor indices
                n1, n2, n3, n4 = self._neighbors[i, j]
                # Sum over nearest neighbors using precomputed indices
                nn_sum = (self.lattice[n1[0], n1[1]] +
                         self.lattice[n2[0], n2[1]] +
                         self.lattice[n3[0], n3[1]] +
                         self.lattice[n4[0], n4[1]])
                energy += -spin * nn_sum
        return energy / 2  # Each pair is counted twice
    
    def _delta_energy(self, i: int, j: int) -> float:
        """
        Calculate the energy change if spin (i,j) were to be flipped.
        
        Args:
            i: Row index of the spin
            j: Column index of the spin
            
        Returns:
            float: Energy change if the spin were flipped
        """
        spin = self.lattice[i, j]
        # Get precomputed neighbor indices
        n1, n2, n3, n4 = self._neighbors[i, j]
        # Sum over nearest neighbors using precomputed indices
        nn_sum = (self.lattice[n1[0], n1[1]] +
                 self.lattice[n2[0], n2[1]] +
                 self.lattice[n3[0], n3[1]] +
                 self.lattice[n4[0], n4[1]])
        return 2 * spin * nn_sum
    
    def step(self) -> None:
        """Perform one Monte Carlo step (attempt to flip size^2 random spins)."""
        for _ in range(self.size * self.size):
            # Randomly select a spin
            i, j = np.random.randint(0, self.size, 2)
            
            # Calculate energy difference if this spin were to flip
            dE = self._delta_energy(i, j)
            
            # Metropolis acceptance criterion
            if dE <= 0 or np.random.random() < np.exp(-dE / self.temperature):
                self.lattice[i, j] *= -1  # Flip the spin
                self.energy += dE
                self.magnetization += 2 * self.lattice[i, j]
    
    def simulate(self, steps: int, burn_in: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the simulation for a given number of steps.
        
        Args:
            steps: Number of Monte Carlo steps to run
            burn_in: Number of initial steps to discard
            
        Returns:
            Tuple of (energies, magnetizations) arrays of length `steps`
        """
        energies = np.zeros(steps)
        magnetizations = np.zeros(steps)
        
        # Burn-in period
        for _ in range(burn_in):
            self.step()
        
        # Main simulation
        for t in range(steps):
            self.step()
            energies[t] = self.energy
            magnetizations[t] = self.magnetization
            
        return energies, magnetizations
