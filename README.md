# Ising Model Simulation and Analysis

This project simulates the 2D and 3D Ising models and provides tools for analyzing the simulation data.

## Project Structure

The project is organized into three main directories:

- `2d_ising_model/`: Contains the C++ source code for the 2D Ising model simulation.
  - `ising_2d.cpp`: Main source file for the 2D simulation.
  - `ising_2d.h`: Header file for the 2D simulation.
  - `2d_ising_pseudocode.txt`: Pseudocode for the 2D model implementation.

- `3d_ising_model/`: Contains the C++ source code for the 3D Ising model simulation.
  - `ising_3d.cpp`: Main source file for the 3D simulation.
  - `ising_3d.h`: Header file for the 3D simulation.
  - `3d_ising_pseudocode.txt`: Pseudocode for the 3D model implementation.

- `analysis/`: Contains Python scripts for data analysis and visualization.
  - `data_analysis.py`: Script for analyzing the output data from the simulations.
  - `visualization.py`: Script for visualizing the simulation results.
  - `visualization_pseudocode.txt`: Pseudocode for the visualization implementation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python implementation of the 2D Ising model using the Metropolis-Hastings algorithm. This package provides tools for simulating the Ising model, visualizing the results, and analyzing the thermodynamic properties of magnetic systems.

## Table of Contents
- [Physics Background](#physics-background)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Physics Background

The Ising model is a mathematical model of ferromagnetism in statistical mechanics. It consists of discrete variables called spins that can be in one of two states (+1 or -1), arranged in a lattice where each spin interacts with its nearest neighbors.

### Key Concepts

- **Spins**: Atomic magnetic moments that can align either up (+1) or down (-1)
- **Lattice**: A 2D grid where each site contains one spin
- **Hamiltonian**: The energy of the system is given by:
  
  $H = -J \sum_{\langle i,j \rangle} s_i s_j - h \sum_i s_i$
  
  where $J$ is the interaction strength, $h$ is the external magnetic field, and $\langle i,j \rangle$ denotes nearest-neighbor pairs.

- **Phase Transition**: At the critical temperature $T_c \approx 2.269$ (in units where $J/k_B = 1$), the system undergoes a second-order phase transition from an ordered to a disordered phase.
- **Order Parameter**: The magnetization per spin serves as the order parameter.

### Physical Significance

The Ising model is a cornerstone of statistical mechanics because:
- It's one of the simplest models showing a phase transition
- It can be solved exactly in 1D and 2D
- It exhibits universality - its critical exponents are shared by many physical systems
- It has applications in modeling ferromagnetism, lattice gas, binary alloys, and neural networks

## Features

- **Efficient Implementation**:
  - Vectorized operations using NumPy
  - Precomputed neighbor indices for optimal performance
  - Parallel processing support for parameter sweeps

- **Visualization**:
  - Interactive lattice visualization
  - Real-time plotting of thermodynamic quantities
  - Animation support for observing dynamics
  - Phase transition analysis tools

- **Physics**:
  - Calculation of key observables:
    - Energy and specific heat
    - Magnetization and susceptibility
    - Correlation functions
  - Support for different boundary conditions
  - Critical temperature estimation

- **Usability**:
  - Clean, object-oriented API
  - Comprehensive documentation
  - Example scripts for common use cases
  - Unit tests with good coverage

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ising_model.git
   cd ising_model
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Quick Start

### Installation

1. Install from PyPI:
   ```bash
   pip install ising-model-simulator
   ```

2. Or install from source:
   ```bash
   git clone https://github.com/yourusername/ising_model.git
   cd ising_model
   pip install -e .
   ```

### Basic Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from ising_model import IsingModel, plot_observables

# Initialize and run simulation
model = IsingModel(size=100, temperature=2.0)
energies, magnetizations = model.simulate(steps=1000, burn_in=100)

# Visualize results
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.imshow(model.lattice, cmap='binary')
plt.title('Final State')

plt.subplot(122)
plot_observables(energies, magnetizations, temperature=2.0)
plt.tight_layout()
plt.show()
```

## Examples

### 1. Phase Transition Visualization

```python
from ising_model import IsingModel
import numpy as np
import matplotlib.pyplot as plt

# Temperature range across the critical point
temps = np.linspace(1.5, 3.0, 20)
final_mags = []

for T in temps:
    model = IsingModel(size=50, temperature=T)
    energies, mags = model.simulate(steps=1000, burn_in=500)
    final_mags.append(np.mean(np.abs(mags[-100:])) / (50*50))

plt.figure(figsize=(8, 5))
plt.plot(temps, final_mags, 'o-', label='Simulation')
plt.axvline(x=2.269, color='r', linestyle='--', label='Theoretical $T_c$')
plt.xlabel('Temperature')
plt.ylabel('Average |Magnetization| per spin')
plt.title('Phase Transition in 2D Ising Model')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. Domain Wall Dynamics

```python
from ising_model import IsingModel
from ising_model.utils import create_domain_wall
import matplotlib.pyplot as plt

# Create initial state with domain wall
initial_state = create_domain_wall(100, orientation='vertical')

# Run simulation
model = IsingModel(size=100, temperature=1.5, initial_state=initial_state)
energies, magnetizations = model.simulate(steps=2000, burn_in=500)

# Visualize
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(initial_state, cmap='binary')
plt.title('Initial State')
plt.axis('off')

plt.subplot(122)
plt.imshow(model.lattice, cmap='binary')
plt.title('Final State')
plt.axis('off')
plt.tight_layout()
plt.show()
```

### 3. Parallel Parameter Sweep

```python
from ising_model.parallel import batch_sweep
import matplotlib.pyplot as plt

# Define parameter grid
params = {
    'size': [50],
    'temperature': np.linspace(1.0, 3.5, 20),
    'steps': [1000],
    'burn_in': [500],
    'initial_state': ['random']
}

# Run parallel simulation
results = batch_sweep(params, max_workers=4)

# Plot results
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(results['param_temperature'], results['mean_magnetization'], 'o-')
plt.xlabel('Temperature')
plt.ylabel('|Magnetization| per spin')
plt.grid(True)

plt.subplot(122)
plt.plot(results['param_temperature'], results['specific_heat'], 's-')
plt.xlabel('Temperature')
plt.ylabel('Specific Heat')
plt.grid(True)
plt.tight_layout()
plt.show()
```

## API Reference

### Core Classes

#### `IsingModel`

```python
IsingModel(size=100, temperature=2.269, initial_state=None)
```

The main class for running Ising model simulations.

**Parameters:**
- `size` (int): Size of the square lattice (default: 100)
- `temperature` (float): Temperature in units of J/k_B (default: 2.269, the critical temperature)
- `initial_state` (np.ndarray, optional): Initial state of the lattice (default: random)

**Attributes:**
- `lattice` (np.ndarray): Current spin configuration
- `energy` (float): Current energy of the system
- `magnetization` (float): Current total magnetization

**Methods:**

##### `simulate(steps, burn_in=0)`
Run the Monte Carlo simulation.
- `steps` (int): Number of Monte Carlo steps
- `burn_in` (int): Number of initial steps to discard (default: 0)
- **Returns:** Tuple of (energies, magnetizations)

##### `step()`
Perform one Monte Carlo step (attempt to flip size² random spins).

### Simulation Module

#### `plot_observables(energies, magnetizations, temperature)`
Plot energy and magnetization over time.

#### `plot_lattice(lattice, title="Ising Model")`
Plot the current state of the lattice.

#### `class IsingAnimation`
Create animations of the Ising model simulation.

### Utils Module

#### `create_domain_wall(size, orientation='horizontal')`
Create initial state with a domain wall.

#### `create_checkerboard(size)`
Create checkerboard initial state.

#### `calculate_observables(energies, magnetizations, temperature, equilibration_time=None)`
Calculate thermodynamic observables from simulation data.

### Parallel Module

#### `batch_sweep(parameters, max_workers=None, **kwargs)`
Run parameter sweeps in parallel.

#### `run_parallel_simulations(parameter_sets, max_workers=None, **kwargs)`
Run multiple simulations in parallel.

## Running Tests

To run the test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# With coverage report
pytest --cov=ising_model tests/
```

## Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and ensure all tests pass.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Daniel V. Schroeder, *An Introduction to Thermal Physics* (2000)
2. M. E. J. Newman and G. T. Barkema, *Monte Carlo Methods in Statistical Physics* (1999)
3. K. Binder and D. W. Heermann, *Monte Carlo Simulation in Statistical Physics* (2010)
4. [The Ising Model: A Window to Understanding Phase Transitions](https://www.compadre.org/PICUP/exercises/Exercise.cfm?I=241&A=IsingModel)
5. [2D Ising Model: Theory, Simulation, and Experiments](https://arxiv.org/abs/0803.0217)

## Citation

If you use this code in your research, please consider citing:

```bibtex
@software{ising_model_simulator,
  author = {Your Name},
  title = {2D Ising Model Simulator},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/ising_model}}
}
```