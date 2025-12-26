"""
Optimized energy evaluation and MCMC sampling examples.

Demonstrates:
1. Flexible atom grouping (from element names)
2. In-memory computation (no file I/O)
3. GPU acceleration (optional)
4. Performance comparison
"""

import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.energy_models.cluster_expansion.optimized_calculator import OptimizedEnergyCalculator
from src.energy_models.cluster_expansion.optimized_mcmc import OptimizedMCMCSampler, compare_performance
from src.energy_models.cluster_expansion.gpu_accelerated import print_gpu_info, check_gpu_availability


def example_flexible_atom_grouping():
    """Example 1: Flexible atom grouping."""
    print("\n" + "="*60)
    print("Example 1: Flexible Atom Grouping")
    print("="*60)

    # Define atom groups by element names
    atom_group = [
        ['Sr', 'La'],  # A-site: Sr can be substituted with La
        ['Ti'],        # B-site: Ti only
        ['O']          # O-site: O only
    ]

    print(f"\nAtom grouping:")
    print(f"  A-site: {atom_group[0]}")
    print(f"  B-site: {atom_group[1]}")
    print(f"  O-site: {atom_group[2]}")

    # Initialize calculator with flexible grouping
    calculator = OptimizedEnergyCalculator(
        model_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_model.pkl',
        scaler_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_scaler.pkl',
        cluster_file='src/energy_models/cluster_expansion/energy_parameter/reference_clusters.json',
        atom_group=atom_group,  # ← Flexible grouping!
        use_gpu=False
    )

    poscar_file = 'src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3'

    if os.path.exists(poscar_file):
        energy = calculator.compute_energy(poscar_file)
        print(f"\nEnergy: {energy:.6f} eV")
        print("✓ Atom grouping automatically converted to indices!")
    else:
        print(f"\nWarning: Test file not found")


def example_inplace_computation():
    """Example 2: In-memory computation (no file I/O)."""
    print("\n" + "="*60)
    print("Example 2: In-Memory Computation (Zero File I/O)")
    print("="*60)

    calculator = OptimizedEnergyCalculator(
        model_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_model.pkl',
        scaler_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_scaler.pkl',
        cluster_file='src/energy_models/cluster_expansion/energy_parameter/reference_clusters.json',
        use_gpu=False
    )

    poscar_file = 'src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3'

    if os.path.exists(poscar_file):
        from src.energy_models.cluster_expansion.structure_utils import posreader, create_atom_type_mapping

        # Load structure once
        poscar = posreader(poscar_file)
        positions = np.array(poscar['LattPnt'])
        lattice = np.array(poscar['Base'])
        atom_types = create_atom_type_mapping(poscar)
        atom_ind_group = calculator._get_atom_ind_group(poscar)

        print("\nComputing energy in-memory (no file I/O)...")

        # Time in-memory computation
        start = time.time()
        energy_inplace = calculator.compute_energy_inplace(
            positions, lattice, atom_types, atom_ind_group
        )
        time_inplace = time.time() - start

        print(f"  Energy: {energy_inplace:.6f} eV")
        print(f"  Time: {time_inplace*1000:.2f} ms")

        # Time file-based computation
        start = time.time()
        energy_file = calculator.compute_energy(poscar_file)
        time_file = time.time() - start

        print(f"\nFile-based computation (for comparison):")
        print(f"  Energy: {energy_file:.6f} eV")
        print(f"  Time: {time_file*1000:.2f} ms")

        print(f"\n✓ In-memory is {time_file/time_inplace:.1f}x faster!")
    else:
        print(f"\nWarning: Test file not found")


def example_optimized_mcmc():
    """Example 3: Optimized MCMC (no file I/O)."""
    print("\n" + "="*60)
    print("Example 3: Optimized MCMC (Zero File I/O)")
    print("="*60)

    calculator = OptimizedEnergyCalculator(
        model_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_model.pkl',
        scaler_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_scaler.pkl',
        cluster_file='src/energy_models/cluster_expansion/energy_parameter/reference_clusters.json',
        use_gpu=False
    )

    sampler = OptimizedMCMCSampler(
        energy_calculator=calculator,
        temperature=1000.0,
        swap_types=[(0, 2)],  # A-site swaps
        random_seed=42
    )

    poscar_file = 'src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3'

    if os.path.exists(poscar_file):
        print("\nRunning optimized MCMC (100 steps)...")
        print("  - No file I/O during sampling")
        print("  - All operations in memory")

        trajectory = sampler.run(
            initial_poscar_file=poscar_file,
            n_steps=100,
            output_dir=None,  # No file output during run
            save_interval=50,
            verbose=True
        )

        print(f"\n✓ MCMC completed!")
        print(f"  Trajectory length: {len(trajectory)}")
        print(f"  Final energy: {trajectory[-1][0]:.6f} eV")
        print(f"  Acceptance rate: {sampler.get_acceptance_rate():.3f}")
    else:
        print(f"\nWarning: Test file not found")


def example_performance_comparison():
    """Example 4: Performance comparison."""
    print("\n" + "="*60)
    print("Example 4: Performance Comparison")
    print("="*60)

    calculator = OptimizedEnergyCalculator(
        model_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_model.pkl',
        scaler_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_scaler.pkl',
        cluster_file='src/energy_models/cluster_expansion/energy_parameter/reference_clusters.json',
        use_gpu=False
    )

    poscar_file = 'src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3'

    if os.path.exists(poscar_file):
        results = compare_performance(calculator, poscar_file, n_test_steps=20)

        print("\n✓ Optimized version is significantly faster!")
        print(f"  Main improvement: Eliminated file I/O bottleneck")
    else:
        print(f"\nWarning: Test file not found")


def example_gpu_acceleration():
    """Example 5: GPU acceleration (if available)."""
    print("\n" + "="*60)
    print("Example 5: GPU Acceleration")
    print("="*60)

    # Check GPU availability
    print_gpu_info()

    gpu_available = check_gpu_availability()

    if gpu_available['cupy'] or gpu_available['torch_cuda']:
        print("\n✓ GPU acceleration is available!")

        calculator_gpu = OptimizedEnergyCalculator(
            model_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_model.pkl',
            scaler_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_scaler.pkl',
            cluster_file='src/energy_models/cluster_expansion/energy_parameter/reference_clusters.json',
            use_gpu=True,  # ← Enable GPU
            gpu_backend='auto'
        )

        print("\nGPU-accelerated calculator initialized:")
        print(calculator_gpu.get_gpu_info())

        # Test with larger batch for GPU benefit
        poscar_file = 'src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3'

        if os.path.exists(poscar_file):
            print("\nTesting GPU batch processing...")

            # Create larger batch (same file for demo)
            batch_files = [poscar_file] * 64

            start = time.time()
            energies = calculator_gpu.compute_energy_batch(batch_files, batch_size=64)
            time_gpu = time.time() - start

            print(f"  Processed {len(batch_files)} structures")
            print(f"  Time: {time_gpu:.3f} seconds")
            print(f"  Per structure: {time_gpu/len(batch_files)*1000:.1f} ms")
    else:
        print("\n⚠ GPU not available")
        print("Install CuPy or PyTorch with CUDA for GPU acceleration")


def example_batch_operations():
    """Example 6: Efficient batch operations."""
    print("\n" + "="*60)
    print("Example 6: Batch Operations")
    print("="*60)

    calculator = OptimizedEnergyCalculator(
        model_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_model.pkl',
        scaler_file='src/energy_models/cluster_expansion/energy_parameter/trained_lasso_scaler.pkl',
        cluster_file='src/energy_models/cluster_expansion/energy_parameter/reference_clusters.json',
        use_gpu=False
    )

    poscar_file = 'src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3'

    if os.path.exists(poscar_file):
        # Test batch processing
        batch_sizes = [1, 8, 32, 64]

        print("\nBatch processing performance:")
        print("  (using same file for demonstration)")

        for batch_size in batch_sizes:
            batch_files = [poscar_file] * batch_size

            start = time.time()
            energies = calculator.compute_energy_batch(batch_files, batch_size=batch_size)
            elapsed = time.time() - start

            per_structure = elapsed / batch_size * 1000

            print(f"\n  Batch size {batch_size:3d}:")
            print(f"    Total time: {elapsed:.3f} s")
            print(f"    Per structure: {per_structure:.1f} ms")
    else:
        print(f"\nWarning: Test file not found")


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" Optimized Energy Evaluation & MCMC Sampling Examples")
    print("="*70)

    # Run examples
    example_flexible_atom_grouping()
    example_inplace_computation()
    example_optimized_mcmc()
    example_performance_comparison()

    # GPU examples (commented out - requires GPU)
    # Uncomment if you have GPU available:
    # example_gpu_acceleration()

    example_batch_operations()

    print("\n" + "="*70)
    print(" All examples completed!")
    print("="*70)
