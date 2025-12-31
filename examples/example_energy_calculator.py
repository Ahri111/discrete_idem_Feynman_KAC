"""
Example usage of Optimized Energy Calculator

This demonstrates:
1. Basic energy calculation
2. Batch processing
3. Incremental updates (for Monte Carlo)
4. Random structure generation
"""

import os
import sys
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.energy_models.cluster_expansion.energy_calculator import EnergyCalculator
from src.energy_models.cluster_expansion.random_structure_generator import (
    generate_random_structure, generate_random_structures_batch, composition_from_ratios
)
from src.energy_models.cluster_expansion.structure_utils import posreader, dismatcreate, dismatswap
import copy
import random


def example_1_basic_usage():
    """Example 1: Basic energy calculation"""
    print("=" * 60)
    print("Example 1: Basic Energy Calculation")
    print("=" * 60)

    # Setup paths
    base_dir = 'src/energy_models/cluster_expansion/energy_parameter'
    model_file = os.path.join(base_dir, 'trained_lasso_model.pkl')
    scaler_file = os.path.join(base_dir, 'trained_lasso_scaler.pkl')
    cluster_file = os.path.join(base_dir, 'reference_clusters.json')
    template_file = os.path.join(base_dir, 'POSCAR_ABO3')

    # Define atom groups (adjust based on your system)
    # This should match your POSCAR element ordering
    atom_ind_group = [
        [0],      # A-site: index 0 (e.g., Sr)
        [1, 2],   # B-site: indices 1, 2 (e.g., Ti, Fe)
        [3, 4]    # O-site: indices 3, 4 (e.g., O, VO)
    ]

    # Create calculator
    print("\n[1] Initializing EnergyCalculator...")
    calculator = EnergyCalculator(
        model_file=model_file,
        scaler_file=scaler_file,
        cluster_file=cluster_file,
        atom_ind_group=atom_ind_group
    )

    # Load a structure
    print("\n[2] Loading structure...")
    poscar = posreader(template_file)
    poscar = dismatcreate(poscar)

    # Compute energy
    print("\n[3] Computing energy...")
    start = time.time()
    energy = calculator.compute_energy(poscar)
    elapsed = time.time() - start

    print(f"   Energy: {energy:.6f} eV")
    print(f"   Time: {elapsed*1000:.2f} ms")


def example_2_batch_processing():
    """Example 2: Batch processing with parallel workers"""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)

    base_dir = 'src/energy_models/cluster_expansion/energy_parameter'
    template_file = os.path.join(base_dir, 'POSCAR_ABO3')

    atom_ind_group = [[0], [1, 2], [3, 4]]

    # Create calculator
    calculator = EnergyCalculator(
        model_file=os.path.join(base_dir, 'trained_lasso_model.pkl'),
        scaler_file=os.path.join(base_dir, 'trained_lasso_scaler.pkl'),
        cluster_file=os.path.join(base_dir, 'reference_clusters.json'),
        atom_ind_group=atom_ind_group
    )

    # Generate random structures
    print("\n[1] Generating random structures...")
    composition = {
        'A': {'Sr': 32},
        'B': {'Ti': 24, 'Fe': 8},
        'O': {'O': 88, 'VO': 4}
    }

    n_structures = 16
    structures = generate_random_structures_batch(
        template_file=template_file,
        composition=composition,
        n_samples=n_structures,
        element_names=['Sr', 'Ti', 'Fe', 'O', 'VO'],
        seed=42
    )

    # Batch computation with different worker counts
    for n_workers in [1, 4, 8]:
        print(f"\n[2] Computing energies with {n_workers} workers...")
        start = time.time()
        energies = calculator.compute_energy_batch(structures, n_workers=n_workers)
        elapsed = time.time() - start

        print(f"   Computed {len(energies)} energies")
        print(f"   Total time: {elapsed:.3f} s")
        print(f"   Per structure: {elapsed/len(energies)*1000:.2f} ms")
        print(f"   Energy range: [{energies.min():.4f}, {energies.max():.4f}]")


def example_3_incremental_update():
    """Example 3: Incremental update for Monte Carlo"""
    print("\n" + "=" * 60)
    print("Example 3: Incremental Update (Monte Carlo)")
    print("=" * 60)

    base_dir = 'src/energy_models/cluster_expansion/energy_parameter'
    template_file = os.path.join(base_dir, 'POSCAR_ABO3')

    atom_ind_group = [[0], [1, 2], [3, 4]]

    calculator = EnergyCalculator(
        model_file=os.path.join(base_dir, 'trained_lasso_model.pkl'),
        scaler_file=os.path.join(base_dir, 'trained_lasso_scaler.pkl'),
        cluster_file=os.path.join(base_dir, 'reference_clusters.json'),
        atom_ind_group=atom_ind_group
    )

    # Generate random structure
    composition = {
        'A': {'Sr': 32},
        'B': {'Ti': 24, 'Fe': 8},
        'O': {'O': 88, 'VO': 4}
    }

    poscar = generate_random_structure(
        template_file=template_file,
        composition=composition,
        element_names=['Sr', 'Ti', 'Fe', 'O', 'VO']
    )

    # Initial energy (with caching)
    print("\n[1] Computing initial energy...")
    start = time.time()
    energy_init = calculator.compute_energy(poscar, use_cache=True, structure_id='mc')
    time_full = time.time() - start
    print(f"   Initial energy: {energy_init:.6f} eV")
    print(f"   Time (full): {time_full*1000:.2f} ms")

    # Simulate Monte Carlo steps
    print("\n[2] Simulating 10 Monte Carlo steps...")
    n_steps = 10

    # Define atom ranges for B-site (Ti and Fe)
    b_site_start = 32  # After A-sites
    b_site_end = 64    # Before O-sites

    total_time_full = 0
    total_time_incremental = 0

    for step in range(n_steps):
        # Random swap within B-site
        idx1 = random.randint(b_site_start, b_site_end - 1)
        idx2 = random.randint(b_site_start, b_site_end - 1)
        while idx1 == idx2:
            idx2 = random.randint(b_site_start, b_site_end - 1)

        # Swap atoms
        poscar['LattPnt'][idx1], poscar['LattPnt'][idx2] = \
            poscar['LattPnt'][idx2], poscar['LattPnt'][idx1]
        poscar['dismat'] = dismatswap(poscar['dismat'], idx1, idx2)

        # Method 1: Full recomputation
        start = time.time()
        energy_full = calculator.compute_energy(poscar, use_cache=False)
        t_full = time.time() - start
        total_time_full += t_full

        # Method 2: Incremental update
        start = time.time()
        energy_incr = calculator.compute_energy_incremental(
            poscar, idx1, idx2, structure_id='mc'
        )
        t_incr = time.time() - start
        total_time_incremental += t_incr

        # Verify they match
        if abs(energy_full - energy_incr) > 1e-6:
            print(f"   WARNING: Energy mismatch at step {step}")

    print(f"\n[3] Results after {n_steps} steps:")
    print(f"   Full recomputation: {total_time_full*1000:.2f} ms total, {total_time_full/n_steps*1000:.2f} ms/step")
    print(f"   Incremental update: {total_time_incremental*1000:.2f} ms total, {total_time_incremental/n_steps*1000:.2f} ms/step")
    print(f"   Speedup: {total_time_full/total_time_incremental:.1f}x")


def example_4_random_structures():
    """Example 4: Generate random structures for diffusion xT"""
    print("\n" + "=" * 60)
    print("Example 4: Random Structure Generation (Diffusion xT)")
    print("=" * 60)

    base_dir = 'src/energy_models/cluster_expansion/energy_parameter'
    template_file = os.path.join(base_dir, 'POSCAR_ABO3')

    # Method 1: Direct composition
    print("\n[1] Method 1: Direct composition")
    composition = {
        'A': {'Sr': 24, 'La': 8},
        'B': {'Ti': 24, 'Fe': 8},
        'O': {'O': 88, 'VO': 4}
    }

    structure = generate_random_structure(
        template_file=template_file,
        composition=composition,
        element_names=['Sr', 'Ti', 'La', 'Fe', 'O', 'VO']
    )

    print(f"   Generated structure:")
    print(f"   Elements: {structure['EleName']}")
    print(f"   Counts: {structure['AtomNum']}")
    print(f"   Total: {structure['AtomSum']}")

    # Method 2: From ratios
    print("\n[2] Method 2: From ratios")
    site_ratios = {
        'A': {'Sr': 3, 'La': 1},    # 3:1 ratio
        'B': {'Ti': 3, 'Fe': 1},    # 3:1 ratio
        'O': {'O': 22, 'VO': 1}     # 22:1 ratio
    }

    total_atoms = {'A': 32, 'B': 32, 'O': 92}

    composition_from_ratio = composition_from_ratios(site_ratios, total_atoms)
    print(f"   Computed composition: {composition_from_ratio}")

    # Generate batch
    print("\n[3] Generating batch of 100 structures...")
    structures = generate_random_structures_batch(
        template_file=template_file,
        composition=composition,
        n_samples=100,
        element_names=['Sr', 'Ti', 'La', 'Fe', 'O', 'VO'],
        seed=42
    )

    print(f"   Generated {len(structures)} structures")
    print(f"   These can be used as xT in diffusion model")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Optimized Energy Calculator Examples")
    print("="*60)

    try:
        example_1_basic_usage()
        example_2_batch_processing()
        example_3_incremental_update()
        example_4_random_structures()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
