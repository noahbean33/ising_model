"""
Utility functions for the Ising Model simulation.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import json
import os

def save_simulation(
    lattice: np.ndarray,
    energies: np.ndarray,
    magnetizations: np.ndarray,
    parameters: Dict[str, Any],
    directory: str = 'results',
    prefix: str = 'ising'
) -> str:
    """
    Save simulation results to files.
    
    Args:
        lattice: Final lattice configuration
        energies: Array of energy values
        magnetizations: Array of magnetization values
        parameters: Dictionary of simulation parameters
        directory: Directory to save results in
        prefix: Prefix for output filenames
        
    Returns:
        Path to the directory containing the saved files
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save lattice
    np.save(os.path.join(directory, f'{prefix}_lattice.npy'), lattice)
    
    # Save time series data
    np.savez(
        os.path.join(directory, f'{prefix}_data.npz'),
        energies=energies,
        magnetizations=magnetizations
    )
    
    # Save parameters
    with open(os.path.join(directory, f'{prefix}_params.json'), 'w') as f:
        json.dump(parameters, f, indent=2)
    
    return os.path.abspath(directory)

def load_simulation(directory: str, prefix: str = 'ising') -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load simulation results from files.
    
    Args:
        directory: Directory containing the saved files
        prefix: Prefix used when saving the files
        
    Returns:
        Tuple of (lattice, energies, magnetizations, parameters)
    """
    # Load lattice
    lattice = np.load(os.path.join(directory, f'{prefix}_lattice.npy'))
    
    # Load time series data
    data = np.load(os.path.join(directory, f'{prefix}_data.npz'))
    energies = data['energies']
    magnetizations = data['magnetizations']
    
    # Load parameters
    with open(os.path.join(directory, f'{prefix}_params.json'), 'r') as f:
        parameters = json.load(f)
    
    return lattice, energies, magnetizations, parameters

def calculate_observables(
    energies: np.ndarray,
    magnetizations: np.ndarray,
    temperature: float,
    equilibration_time: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate various observables from simulation data.
    
    Args:
        energies: Array of energy values
        magnetizations: Array of magnetization values
        temperature: Temperature of the simulation
        equilibration_time: Number of initial steps to discard for equilibration
        
    Returns:
        Dictionary containing the calculated observables
    """
    if equilibration_time is not None:
        energies = energies[equilibration_time:]
        magnetizations = magnetizations[equilibration_time:]
    
    # Calculate mean and variance
    mean_energy = np.mean(energies)
    mean_magnetization = np.mean(np.abs(magnetizations))
    
    # Calculate specific heat and susceptibility
    n = len(energies)
    specific_heat = (np.mean(energies**2) - mean_energy**2) / (temperature**2)
    susceptibility = (np.mean(magnetizations**2) - mean_magnetization**2) / temperature
    
    return {
        'mean_energy': mean_energy,
        'mean_magnetization': mean_magnetization,
        'specific_heat': specific_heat,
        'susceptibility': susceptibility,
        'energy_std': np.std(energies),
        'magnetization_std': np.std(magnetizations)
    }

def create_domain_wall(size: int, orientation: str = 'vertical') -> np.ndarray:
    """
    Create a lattice with a domain wall.
    
    Args:
        size: Size of the lattice
        orientation: 'vertical' or 'horizontal' domain wall
        
    Returns:
        2D array with a domain wall
    """
    lattice = np.ones((size, size))
    half = size // 2
    
    if orientation == 'vertical':
        lattice[:, half:] = -1
    elif orientation == 'horizontal':
        lattice[half:, :] = -1
    else:
        raise ValueError("Orientation must be 'vertical' or 'horizontal'")
    
    return lattice

def create_checkerboard(size: int) -> np.ndarray:
    """
    Create a checkerboard pattern.
    
    Args:
        size: Size of the lattice (must be even)
        
    Returns:
        2D array with a checkerboard pattern
    """
    if size % 2 != 0:
        raise ValueError("Size must be even for checkerboard pattern")
    
    return np.kron([[1, -1], [-1, 1]], np.ones((size//2, size//2)))
