# Generic Pseudocode for 2D Ising Model Simulation

This document outlines the generic pseudocode for simulating a 2-dimensional Ising model. This can be used as a blueprint for implementation in any programming language.

## Data Structures

-   `lattice`: A 2D array (matrix) of size `L x L` to store spin states (+1 for up, -1 for down).
-   `L`: Integer, the dimension of the square lattice.
-   `T`: Float, the temperature of the system.
-   `J`: Float, the coupling constant (interaction strength), typically set to 1.0.

## Core Procedures & Functions

### FUNCTION initialize_lattice(L)
    // Creates and returns a 2D lattice with random spin orientations.
    CREATE a 2D array `lattice` of size `L x L`.
    FOR i FROM 0 TO L-1
        FOR j FROM 0 TO L-1
            IF RANDOM_FLOAT(0, 1) < 0.5 THEN
                lattice[i][j] = 1
            ELSE
                lattice[i][j] = -1
            END IF
        END FOR
    END FOR
    RETURN `lattice`
END FUNCTION

### FUNCTION compute_energy_change(lattice, i, j)
    // Calculates the change in energy if the spin at `(i, j)` is flipped.
    spin = lattice[i][j]
    // Apply periodic boundary conditions
    top_neighbor = lattice[(i - 1 + L) % L][j]
    bottom_neighbor = lattice[(i + 1) % L][j]
    left_neighbor = lattice[i][(j - 1 + L) % L]
    right_neighbor = lattice[i][(j + 1) % L]
    
    sum_of_neighbors = top_neighbor + bottom_neighbor + left_neighbor + right_neighbor
    delta_E = 2 * J * spin * sum_of_neighbors
    RETURN delta_E
END FUNCTION

### PROCEDURE metropolis_step(lattice, T)
    // Performs a single Metropolis update step.
    
    // 1. Select a random spin to consider flipping.
    random_i = RANDOM_INTEGER(0, L-1)
    random_j = RANDOM_INTEGER(0, L-1)
    
    // 2. Calculate the energy change that would result from the flip.
    delta_E = compute_energy_change(lattice, random_i, random_j)
    
    // 3. Decide whether to accept the flip.
    IF delta_E < 0 THEN
        // Always accept if it lowers the energy.
        lattice[random_i][random_j] = -lattice[random_i][random_j]
    ELSE
        // Accept with probability exp(-delta_E / T) if it raises the energy.
        IF RANDOM_FLOAT(0, 1) < EXP(-delta_E / T) THEN
            lattice[random_i][random_j] = -lattice[random_i][random_j]
        END IF
    END IF
END PROCEDURE

## Main Simulation Loop

### PROCEDURE run_simulation(L, T, num_sweeps, equilibration_sweeps)
    // 1. Initialization
    lattice = initialize_lattice(L)
    
    // 2. Equilibration Phase
    FOR sweep FROM 0 TO equilibration_sweeps-1
        FOR step FROM 0 TO (L*L)-1  // One sweep = L*L Metropolis steps
            metropolis_step(lattice, T)
        END FOR
    END FOR
    
    // 3. Measurement Phase
    FOR sweep FROM 0 TO num_sweeps-1
        FOR step FROM 0 TO (L*L)-1
            metropolis_step(lattice, T)
        END FOR
        
        // After each sweep, measure and record observables.
        current_energy = COMPUTE_TOTAL_ENERGY(lattice)
        current_magnetization = COMPUTE_TOTAL_MAGNETIZATION(lattice)
        
        SAVE_DATA(sweep, current_energy, current_magnetization)
    END FOR
END PROCEDURE
