"""
Example script for energy evaluation and MCMC sampling.

Demonstrates:
1. Single structure energy evaluation
2. Batch energy evaluation
3. MCMC sampling (single structure)
4. MCMC sampling (batch)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.energy_models.cluster_expansion.energy_calculator import EnergyCalculator
from src.energy_models.cluster_expansion.mcmc_sampler import MCMCSampler
from src.energy_models.oracles import ClusterExpansionOracle


def example_single_energy_evaluation():
    """Example 1: Single structure energy evaluation."""
    print("\n" + "="*60)
    print("Example 1: Single Structure Energy Evaluation")
    print("="*60)

    # Initialize energy calculator
    calculator = EnergyCalculator(
        model_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_model.pkl',
        scaler_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_scaler.pkl',
        cluster_file='src/energy_models/cluster_expansion/energy_parameter/reference_clusters.json'
    )

    # Compute energy for test structure
    poscar_file = 'src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3'

    if os.path.exists(poscar_file):
        energy = calculator.compute_energy(poscar_file)
        print(f"\nStructure: {poscar_file}")
        print(f"Formation energy: {energy:.6f} eV")
    else:
        print(f"\nWarning: Test file {poscar_file} not found")


def example_batch_energy_evaluation():
    """Example 2: Batch energy evaluation."""
    print("\n" + "="*60)
    print("Example 2: Batch Energy Evaluation")
    print("="*60)

    # Initialize energy calculator
    calculator = EnergyCalculator(
        model_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_model.pkl',
        scaler_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_scaler.pkl',
        cluster_file='src/energy_models/cluster_expansion/energy_parameter/reference_clusters.json'
    )

    # Example: Evaluate multiple structures (same file for demo)
    poscar_file = 'src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3'

    if os.path.exists(poscar_file):
        poscar_files = [poscar_file] * 5  # Demo with same file

        print(f"\nEvaluating {len(poscar_files)} structures with batch_size=2...")
        energies = calculator.compute_energy_batch(poscar_files, batch_size=2)

        print(f"\nResults:")
        for i, energy in enumerate(energies):
            if energy is not None:
                print(f"  Structure {i+1}: {energy:.6f} eV")
            else:
                print(f"  Structure {i+1}: Failed")
    else:
        print(f"\nWarning: Test file {poscar_file} not found")


def example_oracle_usage():
    """Example 3: Using ClusterExpansionOracle."""
    print("\n" + "="*60)
    print("Example 3: ClusterExpansionOracle Usage")
    print("="*60)

    # Initialize oracle
    oracle = ClusterExpansionOracle(
        model_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_model.pkl',
        scaler_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_scaler.pkl',
        cluster_file='src/energy_models/cluster_expansion/energy_parameter/reference_clusters.json',
        device='cpu'
    )

    poscar_file = 'src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3'

    if os.path.exists(poscar_file):
        # Single structure
        print("\nSingle structure evaluation:")
        energy_tensor = oracle.compute_energy(poscar_file)
        print(f"Energy tensor: {energy_tensor}")

        # Batch evaluation
        print("\nBatch evaluation:")
        energy_batch = oracle.compute_energy_batch([poscar_file] * 3, batch_size=2)
        print(f"Energy batch: {energy_batch}")
    else:
        print(f"\nWarning: Test file {poscar_file} not found")


def example_mcmc_sampling():
    """Example 4: MCMC sampling."""
    print("\n" + "="*60)
    print("Example 4: MCMC Sampling")
    print("="*60)

    # Initialize energy calculator
    calculator = EnergyCalculator(
        model_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_model.pkl',
        scaler_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_scaler.pkl',
        cluster_file='src/energy_models/cluster_expansion/energy_parameter/reference_clusters.json'
    )

    # Initialize MCMC sampler
    sampler = MCMCSampler(
        energy_calculator=calculator,
        temperature=1000.0,  # 1000 K
        swap_types=[(0, 2)],  # Allow swapping between A-site types
        random_seed=42
    )

    poscar_file = 'src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3'

    if os.path.exists(poscar_file):
        print(f"\nRunning MCMC sampling:")
        print(f"  Initial structure: {poscar_file}")
        print(f"  Temperature: {sampler.temperature} K")
        print(f"  Number of steps: 100")

        # Run MCMC
        trajectory = sampler.run(
            initial_poscar_file=poscar_file,
            n_steps=100,
            output_dir='mcmc_output',
            save_interval=50,
            verbose=True
        )

        # Plot energy history
        plot_energy_history(sampler.energy_history, 'mcmc_energy_history.png')

        print(f"\nTrajectory length: {len(trajectory)}")
        print(f"Final energy: {trajectory[-1][0]:.6f} eV")
    else:
        print(f"\nWarning: Test file {poscar_file} not found")


def example_mcmc_batch():
    """Example 5: Batch MCMC sampling."""
    print("\n" + "="*60)
    print("Example 5: Batch MCMC Sampling")
    print("="*60)

    # Initialize energy calculator
    calculator = EnergyCalculator(
        model_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_model.pkl',
        scaler_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_scaler.pkl',
        cluster_file='src/energy_models/cluster_expansion/energy_parameter/reference_clusters.json'
    )

    # Initialize MCMC sampler
    sampler = MCMCSampler(
        energy_calculator=calculator,
        temperature=1000.0,
        random_seed=42
    )

    poscar_file = 'src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3'

    if os.path.exists(poscar_file):
        # Run batch MCMC (demo with same file)
        trajectories = sampler.run_batch(
            initial_poscar_files=[poscar_file] * 2,
            n_steps=50,
            output_dirs=['mcmc_batch_1', 'mcmc_batch_2'],
            save_interval=25,
            verbose=True
        )

        print(f"\nCompleted {len(trajectories)} MCMC runs")
    else:
        print(f"\nWarning: Test file {poscar_file} not found")


def plot_energy_history(energy_history, output_file='energy_history.png'):
    """Plot MCMC energy history."""
    plt.figure(figsize=(10, 6))
    plt.plot(energy_history, linewidth=0.5, alpha=0.7)
    plt.xlabel('MCMC Step')
    plt.ylabel('Formation Energy (eV)')
    plt.title('MCMC Energy History')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nEnergy history plot saved to: {output_file}")


if __name__ == '__main__':
    # Run examples
    example_single_energy_evaluation()
    example_batch_energy_evaluation()
    example_oracle_usage()

    # MCMC examples (commented out by default - can be slow)
    # Uncomment to run:
    # example_mcmc_sampling()
    # example_mcmc_batch()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
