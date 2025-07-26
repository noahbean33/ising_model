# Generic Pseudocode for Visualization

This document outlines the generic pseudocode for visualizing the data produced by the Ising model simulations. This is intended to be implemented in Python.

## Core Procedures

### PROCEDURE plot_observables(data_file)
    // Reads and plots time-series data like energy and magnetization.
    
    // 1. Load data from the specified file.
    data = READ_TABULAR_DATA(data_file) // Expects columns: sweep, energy, magnetization
    
    // 2. Create a plot window with two subplots.
    CREATE plot_window with two panels: panel_energy, panel_magnetization
    
    // 3. Plot Energy vs. Sweep.
    PLOT data.sweep vs. data.energy on panel_energy
    SET_X_LABEL(panel_energy, "Monte Carlo Sweeps")
    SET_Y_LABEL(panel_energy, "Total Energy")
    SET_TITLE(panel_energy, "Energy vs. Sweeps")
    
    // 4. Plot Magnetization vs. Sweep.
    PLOT data.sweep vs. data.magnetization on panel_magnetization
    SET_X_LABEL(panel_magnetization, "Monte Carlo Sweeps")
    SET_Y_LABEL(panel_magnetization, "Total Magnetization")
    SET_TITLE(panel_magnetization, "Magnetization vs. Sweeps")
    
    // 5. Display the plot.
    SHOW_PLOT()
END PROCEDURE

### PROCEDURE visualize_2d_lattice(lattice_file)
    // Renders a 2D lattice configuration from a data file.
    
    // 1. Load the 2D lattice data.
    lattice_2d = READ_2D_ARRAY(lattice_file)
    
    // 2. Create a heatmap or image plot.
    CREATE plot_window
    DISPLAY_AS_HEATMAP(lattice_2d, colors={-1: color_down, 1: color_up})
    
    // 3. Add plot details.
    SET_TITLE("2D Ising Lattice Configuration")
    ADD_COLOR_LEGEND(labels=["Spin Down (-1)", "Spin Up (+1)"])
    
    // 4. Display the plot.
    SHOW_PLOT()
END PROCEDURE

### PROCEDURE visualize_3d_lattice(lattice_file)
    // Renders a 3D lattice configuration from a data file.
    
    // 1. Load the 3D lattice data.
    lattice_3d = READ_3D_ARRAY(lattice_file)
    
    // 2. Create a 3D voxel plot.
    CREATE 3d_plot_window
    FOR each point (i, j, k) in lattice_3d
        IF lattice_3d[i][j][k] == 1 THEN
            DRAW_VOXEL at (i, j, k) with color_up
        ELSE IF lattice_3d[i][j][k] == -1 THEN
            DRAW_VOXEL at (i, j, k) with color_down
        END IF
    END FOR
    
    // 3. Add plot details.
    SET_TITLE("3D Ising Lattice Configuration")
    
    // 4. Display the plot.
    SHOW_PLOT()
END PROCEDURE

## Main Program Logic

BEGIN
    // Example of how the visualization procedures could be called.
    CALL plot_observables("../data/2d_ising_data.csv")
    CALL visualize_2d_lattice("../data/2d_lattice_final_state.dat")
    
    CALL plot_observables("../data/3d_ising_data.csv")
    CALL visualize_3d_lattice("../data/3d_lattice_final_state.dat")
END
