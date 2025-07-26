# Generic Pseudocode for 3D Ising Model Simulation

This document outlines the generic pseudocode for simulating a 3-dimensional Ising model. This can be used as a blueprint for implementation in any programming language.

## Data Structures

-   `lattice`: A 3D array of size `L x L x L` to store spin states (+1 for up, -1 for down).
-   `L`: Integer, the dimension of the cubic lattice.
-   `T`: Float, the temperature of the system.
-   `J`: Float, the coupling constant (interaction strength), typically set to 1.0.

## Core Procedures & Functions

### FUNCTION initialize_lattice(L)
    // Creates and returns a 3D lattice with random spin orientations.
    CREATE a 3D array `lattice` of size `L x L x L`.
    FOR i FROM 0 TO L-1
        FOR j FROM 0 TO L-1
            FOR k FROM 0 TO L-1
                IF RANDOM_FLOAT(0, 1) < 0.5 THEN
                    lattice[i][j][k] = 1
                ELSE
                    lattice[i][j][k] = -1
                END IF
            END FOR
        END FOR
    END FOR
    RETURN `lattice`
END FUNCTION

### FUNCTION compute_energy_change(lattice, i, j, k)
    // Calculates the change in energy if the spin at `(i, j, k)` is flipped.
    spin = lattice[i][j][k]
    // Apply periodic boundary conditions for all 6 neighbors
    neighbor1 = lattice[(i - 1 + L) % L][j][k]
    neighbor2 = lattice[(i + 1) % L][j][k]
    neighbor3 = lattice[i][(j - 1 + L) % L][k]
    neighbor4 = lattice[i][(j + 1) % L][k]
    neighbor5 = lattice[i][j][(k - 1 + L) % L]
    neighbor6 = lattice[i][j][(k + 1) % L]
    
    sum_of_neighbors = neighbor1 + neighbor2 + neighbor3 + neighbor4 + neighbor5 + neighbor6
    delta_E = 2 * J * spin * sum_of_neighbors
    RETURN delta_E
END FUNCTION

### PROCEDURE metropolis_step(lattice, T)
    // Performs a single Metropolis update step.
    
    // 1. Select a random spin to consider flipping.
    random_i = RANDOM_INTEGER(0, L-1)
    random_j = RANDOM_INTEGER(0, L-1)
    random_k = RANDOM_INTEGER(0, L-1)
    
    // 2. Calculate the energy change that would result from the flip.
    delta_E = compute_energy_change(lattice, random_i, random_j, random_k)
    
    // 3. Decide whether to accept the flip.
    IF delta_E < 0 THEN
        // Always accept if it lowers the energy.
        lattice[random_i][random_j][random_k] = -lattice[random_i][random_j][random_k]
    ELSE
        // Accept with probability exp(-delta_E / T) if it raises the energy.
        IF RANDOM_FLOAT(0, 1) < EXP(-delta_E / T) THEN
            lattice[random_i][random_j][random_k] = -lattice[random_i][random_j][random_k]
        END IF
    END IF
END PROCEDURE

## Main Simulation Loop

### PROCEDURE run_simulation(L, T, num_sweeps, equilibration_sweeps)
    // 1. Initialization
    lattice = initialize_lattice(L)
    
    // 2. Equilibration Phase
    FOR sweep FROM 0 TO equilibration_sweeps-1
        FOR step FROM 0 TO (L*L*L)-1  // One sweep = L*L*L Metropolis steps
            metropolis_step(lattice, T)
        END FOR
    END FOR
    
    // 3. Measurement Phase
    FOR sweep FROM 0 TO num_sweeps-1
        FOR step FROM 0 TO (L*L*L)-1
            metropolis_step(lattice, T)
        END FOR
        
        // After each sweep, measure and record observables.
        current_energy = COMPUTE_TOTAL_ENERGY(lattice)
        current_magnetization = COMPUTE_TOTAL_MAGNETIZATION(lattice)
        
        SAVE_DATA(sweep, current_energy, current_magnetization)
    END FOR
END PROCEDURE
