"""
Parallel processing and batch processing utilities for the Ising Model simulation.

This module provides tools for running multiple Ising model simulations in parallel,
with support for batching to manage memory usage during large parameter sweeps.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Generator, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import json
from pathlib import Path
from tqdm import tqdm
import warnings

from .model import IsingModel
from .utils import save_simulation, calculate_observables

# Type aliases
ParameterSet = Dict[str, Any]
ParameterList = List[ParameterSet]
ResultDict = Dict[str, Any]
ResultList = List[ResultDict]


def batch_parameters(
    parameter_sets: ParameterList,
    batch_size: int = None,
    max_workers: int = None
) -> Generator[ParameterList, None, None]:
    """
    Split parameter sets into batches for processing.
    
    Args:
        parameter_sets: List of parameter dictionaries
        batch_size: Number of parameter sets per batch. If None, calculated automatically.
        max_workers: Maximum number of worker processes. Used to calculate batch_size if not provided.
        
    Yields:
        Batches of parameter sets
    """
    if not parameter_sets:
        return []
        
    if batch_size is None:
        if max_workers is None:
            import os
            max_workers = os.cpu_count() or 4
        # Default to 2 batches per worker to balance load and memory
        batch_size = max(1, len(parameter_sets) // (2 * max_workers))
    
    for i in range(0, len(parameter_sets), batch_size):
        yield parameter_sets[i:i + batch_size]


def _run_single_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single simulation with given parameters.
    
    This is a helper function to be used with the parallel processing.
    
    Args:
        params: Dictionary containing simulation parameters
        
    Returns:
        Dictionary containing simulation results and metadata
    """
    try:
        # Extract parameters with defaults
        size = params.get('size', 100)
        temperature = params['temperature']
        steps = params.get('steps', 1000)
        burn_in = params.get('burn_in', 100)
        initial_state = params.get('initial_state')
        
        # Create and run model
        model = IsingModel(size=size, temperature=temperature, initial_state=initial_state)
        start_time = time.time()
        energies, magnetizations = model.simulate(steps=steps, burn_in=burn_in)
        end_time = time.time()
        
        # Calculate observables
        obs = calculate_observables(energies, magnetizations, temperature, 
                                  equilibration_time=burn_in)
        
        # Prepare results
        result = {
            'parameters': params,
            'lattice': model.lattice,
            'energies': energies,
            'magnetizations': magnetizations,
            'observables': obs,
            'execution_time': end_time - start_time,
            'pid': os.getpid(),
            'success': True
        }
        
        # Save results if requested
        if params.get('save_results', False):
            save_dir = params.get('save_dir', 'results')
            save_prefix = params.get('save_prefix', f'ising_T{temperature:.3f}')
            save_simulation(
                lattice=model.lattice,
                energies=energies,
                magnetizations=magnetizations,
                parameters=params,
                directory=save_dir,
                prefix=save_prefix
            )
            result['save_path'] = os.path.abspath(save_dir)
            
        return result
        
    except Exception as e:
        return {
            'parameters': params,
            'error': str(e),
            'success': False,
            'pid': os.getpid()
        }


def run_parallel_simulations(
    parameter_sets: ParameterList,
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    progress: bool = True,
    save_batch_results: bool = False,
    batch_save_dir: str = 'batch_results',
    **executor_kwargs
) -> ResultList:
    """
    Run multiple Ising model simulations in parallel with batch processing.
    
    Args:
        parameter_sets: List of parameter dictionaries for each simulation
        max_workers: Maximum number of worker processes to use. If None, uses os.cpu_count().
        batch_size: Number of simulations to process in each batch. If None, calculated automatically.
        progress: Whether to show a progress bar
        save_batch_results: Whether to save results after each batch
        batch_save_dir: Directory to save batch results
        **executor_kwargs: Additional keyword arguments to pass to ProcessPoolExecutor
        
    Returns:
        List of result dictionaries, one for each simulation
    """
    if not parameter_sets:
        return []
        
    # Create save directory if needed
    if save_batch_results:
        os.makedirs(batch_save_dir, exist_ok=True)
    
    all_results = []
    batch_num = 0
    
    # Process in batches
    for batch in batch_parameters(parameter_sets, batch_size, max_workers):
        batch_results = []
        batch_num += 1
        
        with ProcessPoolExecutor(max_workers=max_workers, **executor_kwargs) as executor:
            # Submit batch tasks
            future_to_params = {
                executor.submit(_run_single_simulation, params): i 
                for i, params in enumerate(batch)
            }
            
            # Set up progress bar
            futures = future_to_params.keys()
            if progress:
                desc = f"Batch {batch_num} ({len(batch)} sims)"
                futures = tqdm(as_completed(future_to_params), total=len(batch), 
                             desc=desc, leave=False)
            
            # Process results as they complete
            for future in futures:
                result = future.result()
                batch_results.append(result)
                
                # Update progress bar description
                if progress and 'parameters' in result and 'temperature' in result['parameters']:
                    temp = result['parameters']['temperature']
                    futures.set_description(f"Batch {batch_num}: T = {temp:.3f}")
        
        # Sort batch results to match input order
        batch_results.sort(key=lambda x: future_to_params.get(x.get('__future__', 0), 0))
        all_results.extend(batch_results)
        
        # Save batch results if requested
        if save_batch_results and batch_results:
            batch_file = os.path.join(batch_save_dir, f'batch_{batch_num:04d}.json')
            _save_batch_results(batch_results, batch_file)
    
    return all_results


def _save_batch_results(results: ResultList, filename: str) -> None:
    """Save batch results to a JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = []
    
    for result in results:
        serialized = {}
        for key, value in result.items():
            if key in ['lattice', 'energies', 'magnetizations'] and hasattr(value, 'tolist'):
                # Convert numpy arrays to lists
                serialized[key] = value.tolist()
            elif key == 'parameters' and 'initial_state' in value and value['initial_state'] is not None:
                # Handle initial_state in parameters
                params = value.copy()
                if hasattr(params['initial_state'], 'tolist'):
                    params['initial_state'] = params['initial_state'].tolist()
                serialized[key] = params
            else:
                serialized[key] = value
        serializable_results.append(serialized)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def temperature_sweep(
    temperatures: Union[List[float], np.ndarray],
    size: int = 50,
    steps: int = 1000,
    burn_in: int = 100,
    initial_state: Optional[np.ndarray] = None,
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    progress: bool = True,
    save_results: bool = False,
    save_dir: str = 'results',
    save_batch_results: bool = False,
    batch_save_dir: str = 'batch_results',
    **simulation_kwargs
) -> ResultList:
    """
    Run simulations at different temperatures in parallel with batch processing.
    
    Args:
        temperatures: List of temperatures to simulate at
        size: Size of the lattice
        steps: Number of Monte Carlo steps per simulation
        burn_in: Number of burn-in steps to discard
        initial_state: Initial state of the lattice (same for all temperatures)
        max_workers: Maximum number of worker processes
        batch_size: Number of simulations per batch. If None, calculated automatically.
        progress: Whether to show a progress bar
        save_results: Whether to save individual simulation results to disk
        save_dir: Directory to save individual results
        save_batch_results: Whether to save batch results to disk
        batch_save_dir: Directory to save batch results
        **simulation_kwargs: Additional keyword arguments to pass to the simulation
        
    Returns:
        List of result dictionaries, one for each temperature
    """
    # Convert numpy array to list if needed
    if isinstance(temperatures, np.ndarray):
        temperatures = temperatures.tolist()
    
    # Prepare parameter sets
    parameter_sets = []
    for i, temp in enumerate(temperatures):
        params = {
            'size': size,
            'temperature': float(temp),
            'steps': steps,
            'burn_in': burn_in,
            'initial_state': initial_state.copy() if initial_state is not None else None,
            'save_results': save_results,
            'save_dir': save_dir,
            'save_prefix': f'ising_T{float(temp):.3f}',
            **simulation_kwargs
        }
        parameter_sets.append(params)
    
    # Run simulations in parallel with batching
    return run_parallel_simulations(
        parameter_sets=parameter_sets,
        max_workers=max_workers,
        batch_size=batch_size,
        progress=progress,
        save_batch_results=save_batch_results,
        batch_save_dir=batch_save_dir
    )


def analyze_phase_transition(
    t_min: float = 1.0,
    t_max: float = 3.5,
    num_points: int = 20,
    batch_size: Optional[int] = None,
    save_batch_results: bool = True,
    batch_save_dir: str = 'phase_transition_batches',
    **kwargs
) -> Dict[str, Any]:
    """
    Perform a temperature sweep to analyze the phase transition with batch processing.
    
    Args:
        t_min: Minimum temperature
        t_max: Maximum temperature
        num_points: Number of temperature points
        batch_size: Number of temperatures to process in each batch
        save_batch_results: Whether to save batch results to disk
        batch_save_dir: Directory to save batch results
        **kwargs: Additional arguments to pass to temperature_sweep
        
    Returns:
        Dictionary containing analysis results
    """
    # Generate temperature points (more points near critical temperature)
    t_critical = 2.269  # Known critical temperature for 2D Ising model
    
    # Create more points near the critical temperature
    t1 = np.linspace(t_min, t_critical * 0.9, num_points // 2, endpoint=False)
    t2 = np.linspace(t_critical * 0.9, t_critical * 1.1, num_points // 4, endpoint=False)
    t3 = np.linspace(t_critical * 1.1, t_max, num_points // 4)
    temperatures = np.concatenate([t1, t2, t3])
    
    # Run simulations with batching
    results = temperature_sweep(
        temperatures=temperatures,
        batch_size=batch_size,
        save_batch_results=save_batch_results,
        batch_save_dir=batch_save_dir,
        **kwargs
    )
    
    # Sort results by temperature (in case they were processed out of order)
    results.sort(key=lambda x: x['parameters']['temperature'])
    
    # Extract observables
    try:
        analysis = {
            'temperatures': np.array([r['parameters']['temperature'] for r in results]),
            'magnetizations': np.array([r['observables']['mean_magnetization'] for r in results]),
            'magnetization_stds': np.array([r['observables']['magnetization_std'] for r in results]),
            'energies': np.array([r['observables']['mean_energy'] for r in results]),
            'energy_stds': np.array([r['observables']['energy_std'] for r in results]),
            'specific_heats': np.array([r['observables']['specific_heat'] for r in results]),
            'susceptibilities': np.array([r['observables']['susceptibility'] for r in results]),
            'results': results,
            'metadata': {
                't_critical': t_critical,
                'num_points': num_points,
                'batch_size': batch_size,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'parameters': kwargs.get('simulation_kwargs', {})
            }
        }
        
        # Save full analysis
        if save_batch_results:
            os.makedirs(batch_save_dir, exist_ok=True)
            analysis_file = os.path.join(batch_save_dir, 'phase_transition_analysis.json')
            
            # Convert numpy arrays to lists for JSON serialization
            serializable = {}
            for key, value in analysis.items():
                if key == 'results':
                    # Skip results as they're already saved in batches
                    continue
                elif hasattr(value, 'tolist'):
                    serializable[key] = value.tolist()
                else:
                    serializable[key] = value
            
            with open(analysis_file, 'w') as f:
                json.dump(serializable, f, indent=2)
        
        return analysis
        
    except KeyError as e:
        raise ValueError(f"Error processing results: {e}. Results may be incomplete or malformed.") from e


def load_phase_transition_results(directory: str) -> Dict[str, Any]:
    """
    Load phase transition results from a directory.
    
    Args:
        directory: Directory containing batch results
        
    Returns:
        Dictionary containing combined analysis results
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all batch files
    batch_files = sorted(directory.glob('batch_*.json'))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {directory}")
    
    # Load all results
    all_results = []
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            batch_results = json.load(f)
            all_results.extend(batch_results)
    
    # Sort by temperature
    all_results.sort(key=lambda x: x['parameters']['temperature'])
    
    # Convert back to numpy arrays
    for result in all_results:
        for key in ['lattice', 'energies', 'magnetizations']:
            if key in result and isinstance(result[key], list):
                result[key] = np.array(result[key])
    
    # Recreate analysis
    analysis = {
        'temperatures': np.array([r['parameters']['temperature'] for r in all_results]),
        'magnetizations': np.array([r['observables']['mean_magnetization'] for r in all_results]),
        'magnetization_stds': np.array([r['observables'].get('magnetization_std', 0) for r in all_results]),
        'energies': np.array([r['observables']['mean_energy'] for r in all_results]),
        'energy_stds': np.array([r['observables'].get('energy_std', 0) for r in all_results]),
        'specific_heats': np.array([r['observables']['specific_heat'] for r in all_results]),
        'susceptibilities': np.array([r['observables']['susceptibility'] for r in all_results]),
        'results': all_results
    }
    
    return analysis
