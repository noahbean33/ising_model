"""
Example of batch processing for parameter sweeps in the Ising Model.

This script demonstrates how to use the batch processing capabilities to efficiently
run multiple simulations with different parameters and analyze the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

from ising_model.parallel import (
    batch_sweep,
    analyze_phase_transition,
    load_phase_transition_results
)
from ising_model.utils import create_domain_wall, create_checkerboard

def run_batch_example():
    """Run a batch processing example with different initial states and temperatures."""
    print("Running batch processing example...")
    
    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"batch_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Define parameter grid
    parameters = {
        'size': [50],
        'temperature': np.linspace(1.0, 3.5, 20).tolist(),
        'steps': [2000],
        'burn_in': [500],
        'initial_state': ['random', 'up', 'domain_vertical', 'checkerboard'],
    }
    
    print(f"Parameter grid will run {np.prod([len(v) for v in parameters.values()])} simulations")
    print(f"Results will be saved to: {output_dir.absolute()}")
    
    # Run batch processing
    start_time = time.time()
    results = batch_sweep(
        parameters=parameters,
        output_dir=str(output_dir),
        max_workers=os.cpu_count(),  # Use all available cores
        batch_size=4,  # Process 4 simulations per batch
        progress=True,
        save_batch_results=True,
        overwrite=True
    )
    
    # Save results summary
    results.to_csv(output_dir / 'results_summary.csv', index=False)
    
    # Analyze and plot results
    analyze_batch_results(results, output_dir)
    
    print(f"Batch processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Results saved to: {output_dir.absolute()}")


def analyze_batch_results(results_df, output_dir):
    """Analyze and plot batch results."""
    import seaborn as sns
    
    print("\nAnalyzing batch results...")
    
    # Create plots directory
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Set plot style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # Plot magnetization vs temperature for each initial state
    plt.figure()
    sns.lineplot(
        data=results_df,
        x='param_temperature',
        y='mean_magnetization',
        hue='param_initial_state',
        marker='o',
        markersize=8,
        linewidth=2
    )
    plt.axvline(x=2.269, color='r', linestyle='--', label='Critical T')
    plt.xlabel('Temperature')
    plt.ylabel('|Magnetization| per spin')
    plt.title('Magnetization vs Temperature by Initial State')
    plt.legend(title='Initial State')
    plt.tight_layout()
    plt.savefig(plots_dir / 'magnetization_vs_temperature.png', dpi=150, bbox_inches='tight')
    
    # Plot energy vs temperature for each initial state
    plt.figure()
    sns.lineplot(
        data=results_df,
        x='param_temperature',
        y='mean_energy',
        hue='param_initial_state',
        marker='o',
        markersize=8,
        linewidth=2
    )
    plt.axvline(x=2.269, color='r', linestyle='--', label='Critical T')
    plt.xlabel('Temperature')
    plt.ylabel('Energy per spin')
    plt.title('Energy vs Temperature by Initial State')
    plt.legend(title='Initial State')
    plt.tight_layout()
    plt.savefig(plots_dir / 'energy_vs_temperature.png', dpi=150, bbox_inches='tight')
    
    # Plot specific heat and susceptibility
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.lineplot(
        data=results_df,
        x='param_temperature',
        y='specific_heat',
        hue='param_initial_state',
        ax=ax1,
        marker='o',
        markersize=6,
        linewidth=2
    )
    ax1.axvline(x=2.269, color='r', linestyle='--')
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Specific Heat')
    ax1.set_title('Specific Heat vs Temperature')
    ax1.legend(title='Initial State')
    
    sns.lineplot(
        data=results_df,
        x='param_temperature',
        y='susceptibility',
        hue='param_initial_state',
        ax=ax2,
        marker='o',
        markersize=6,
        linewidth=2
    )
    ax2.axvline(x=2.269, color='r', linestyle='--')
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Magnetic Susceptibility')
    ax2.set_title('Susceptibility vs Temperature')
    ax2.legend(title='Initial State')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'response_functions.png', dpi=150, bbox_inches='tight')
    
    # Save a summary of the results
    with open(plots_dir / 'analysis_summary.txt', 'w') as f:
        f.write("Ising Model Batch Processing Results\n")
        f.write("=" * 30 + "\n\n")
        
        f.write(f"Total simulations: {len(results_df)}\n")
        f.write(f"Temperatures: {results_df['param_temperature'].min():.2f} to {results_df['param_temperature'].max():.2f}\n")
        f.write(f"Lattice size: {results_df['param_size'].iloc[0]}x{results_df['param_size'].iloc[0]}\n")
        f.write(f"Initial states: {', '.join(results_df['param_initial_state'].unique())}\n\n")
        
        # Find critical temperature estimate from specific heat peak
        max_cv_idx = results_df['specific_heat'].idxmax()
        t_c_est = results_df.loc[max_cv_idx, 'param_temperature']
        f.write(f"Estimated critical temperature (from specific heat peak): {t_c_est:.3f} (actual: 2.269)\n")
        
        # Find critical temperature estimate from susceptibility peak
        max_chi_idx = results_df['susceptibility'].idxmax()
        t_c_est_chi = results_df.loc[max_chi_idx, 'param_temperature']
        f.write(f"Estimated critical temperature (from susceptibility peak): {t_c_est_chi:.3f} (actual: 2.269)\n")
    
    print(f"Analysis complete. Plots saved to: {plots_dir.absolute()}")


def run_phase_transition_analysis():
    """Run a detailed phase transition analysis with batch processing."""
    print("\nRunning phase transition analysis...")
    
    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"phase_transition_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Run phase transition analysis with batch processing
    analysis = analyze_phase_transition(
        t_min=1.0,
        t_max=3.5,
        num_points=30,  # More points for better resolution
        size=64,  # Larger lattice for better statistics
        steps=5000,  # More steps for better equilibration
        burn_in=1000,  # More burn-in for larger lattice
        initial_state='random',
        max_workers=os.cpu_count(),
        batch_size=4,
        save_batch_results=True,
        batch_save_dir=str(output_dir / 'batches'),
        progress=True
    )
    
    # Save analysis results
    np.savez(
        output_dir / 'phase_transition.npz',
        temperatures=analysis['temperatures'],
        magnetizations=analysis['magnetizations'],
        energies=analysis['energies'],
        specific_heats=analysis['specific_heats'],
        susceptibilities=analysis['susceptibilities']
    )
    
    # Plot results
    plot_phase_transition(analysis, output_dir)
    
    print(f"Phase transition analysis complete. Results saved to: {output_dir.absolute()}")
    
    return analysis


def plot_phase_transition(analysis, output_dir):
    """Plot phase transition analysis results."""
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Magnetization and Energy
    plt.subplot(2, 1, 1)
    
    # Magnetization (left y-axis)
    ax1 = plt.gca()
    ax1.plot(analysis['temperatures'], analysis['magnetizations'], 'o-', color='tab:blue', label='|Magnetization|')
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('|Magnetization| per spin', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Energy (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(analysis['temperatures'], analysis['energies'], 's-', color='tab:red', label='Energy')
    ax2.set_ylabel('Energy per spin', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Critical temperature line
    ax1.axvline(x=2.269, color='k', linestyle='--', label='Critical T')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Phase Transition: Magnetization and Energy')
    
    # Plot 2: Response Functions
    plt.subplot(2, 1, 2)
    
    # Specific Heat (left y-axis)
    ax3 = plt.gca()
    ax3.plot(analysis['temperatures'], analysis['specific_heats'], 'o-', color='tab:green', label='Specific Heat')
    ax3.set_xlabel('Temperature')
    ax3.set_ylabel('Specific Heat', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    
    # Susceptibility (right y-axis)
    ax4 = ax3.twinx()
    ax4.plot(analysis['temperatures'], analysis['susceptibilities'], 's-', color='tab:purple', label='Susceptibility')
    ax4.set_ylabel('Magnetic Susceptibility', color='tab:purple')
    ax4.tick_params(axis='y', labelcolor='tab:purple')
    
    # Critical temperature line
    ax3.axvline(x=2.269, color='k', linestyle='--', label='Critical T')
    
    # Combine legends
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right')
    
    plt.title('Response Functions')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'phase_transition_analysis.png', dpi=150, bbox_inches='tight')
    
    # Save data for future reference
    np.savez(
        plots_dir / 'phase_transition_data.npz',
        temperatures=analysis['temperatures'],
        magnetizations=analysis['magnetizations'],
        energies=analysis['energies'],
        specific_heats=analysis['specific_heats'],
        susceptibilities=analysis['susceptibilities']
    )


if __name__ == "__main__":
    print("Ising Model Batch Processing Example")
    print("=" * 40)
    
    # Run batch processing example
    run_batch_example()
    
    # Run detailed phase transition analysis
    run_phase_transition_analysis()
    
    print("\nAll examples completed successfully!")
