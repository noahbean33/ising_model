# Generic Pseudocode for 1D Ising Model Simulation

This document outlines the generic pseudocode for simulating a 1-dimensional Ising model. This can be used as a blueprint for implementation in any programming language.

## Data Structures

-   `lattice`: A 1D array of size `L` to store spin states (+1 for up, -1 for down).
-   `L`: Integer, the length of the 1D lattice.
-   `T`: Float, the temperature of the system.
-   `J`: Float, the coupling constant (interaction strength), typically set to 1.0.

## Core Procedures & Functions

### FUNCTION initialize_lattice(L)
    // Creates and returns a 1D lattice with random spin orientations.
    CREATE a 1D array `lattice` of size `L`.
    FOR i FROM 0 TO L-1
        IF RANDOM_FLOAT(0, 1) < 0.5 THEN
            lattice[i] = 1
        ELSE
            lattice[i] = -1
        END IF
    END FOR
    RETURN `lattice`
END FUNCTION

### FUNCTION compute_energy_change(lattice, index)
    // Calculates the change in energy if the spin at `index` is flipped.
    spin = lattice[index]
    left_neighbor = lattice[(index - 1 + L) % L]  // Periodic boundary conditions
    right_neighbor = lattice[(index + 1) % L] // Periodic boundary conditions
    
    delta_E = 2 * J * spin * (left_neighbor + right_neighbor)
    RETURN delta_E
END FUNCTION

### PROCEDURE metropolis_step(lattice, T)
    // Performs a single Metropolis update step.
    
    // 1. Select a random spin to consider flipping.
    random_index = RANDOM_INTEGER(0, L-1)
    
    // 2. Calculate the energy change that would result from the flip.
    delta_E = compute_energy_change(lattice, random_index)
    
    // 3. Decide whether to accept the flip.
    IF delta_E < 0 THEN
        // Always accept if it lowers the energy.
        lattice[random_index] = -lattice[random_index]
    ELSE
        // Accept with probability exp(-delta_E / T) if it raises the energy.
        IF RANDOM_FLOAT(0, 1) < EXP(-delta_E / T) THEN
            lattice[random_index] = -lattice[random_index]
        END IF
    END IF
END PROCEDURE

## Main Simulation Loop

### PROCEDURE run_simulation(L, T, num_sweeps, equilibration_sweeps)
    // 1. Initialization
    lattice = initialize_lattice(L)
    
    // 2. Equilibration Phase (run without measurement to reach a steady state)
    FOR sweep FROM 0 TO equilibration_sweeps-1
        FOR step FROM 0 TO L-1  // One sweep consists of L Metropolis steps
            metropolis_step(lattice, T)
        END FOR
    END FOR
    
    // 3. Measurement Phase
    FOR sweep FROM 0 TO num_sweeps-1
        FOR step FROM 0 TO L-1
            metropolis_step(lattice, T)
        END FOR
        
        // After each sweep, measure and record observables.
        current_energy = COMPUTE_TOTAL_ENERGY(lattice)
        current_magnetization = COMPUTE_TOTAL_MAGNETIZATION(lattice)
        
        SAVE_DATA(sweep, current_energy, current_magnetization)
    END FOR
END PROCEDURE
