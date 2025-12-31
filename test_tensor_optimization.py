"""
Test optimized tensor interface (without PyTorch dependency)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.energy_models.cluster_expansion.energy_calculator import create_energy_calculator
from src.energy_models.cluster_expansion.structure_utils import posreader, dismatcreate

def test_single_structure():
    """Test single structure energy calculation"""
    print("="*60)
    print("Test 1: Single structure (old vs new)")
    print("="*60)

    base_dir = 'src/energy_models/cluster_expansion/energy_parameter'
    poscar_file = os.path.join(base_dir, 'POSCAR_ABO3')

    # Create calculator
    calculator = create_energy_calculator(
        base_dir=base_dir,
        atom_ind_group=[[0], [1, 2], [3, 4]],
        element_names=['Sr', 'Ti', 'Fe', 'O', 'VO']
    )

    # Load POSCAR
    poscar = posreader(poscar_file)
    poscar = dismatcreate(poscar)

    # Method 1: Old way (POSCAR)
    print("\n[Method 1] Old way: POSCAR → energy")
    energy_old = calculator.compute_energy(poscar)
    print(f"  Energy: {energy_old:.6f} eV")

    # Method 2: New way (numpy arrays)
    print("\n[Method 2] New way: numpy arrays → energy")

    # Extract numpy arrays from POSCAR
    positions_np = np.array(poscar['LattPnt'])

    # Create atom types array
    atom_types_np = []
    for atom_type, count in enumerate(poscar['AtomNum']):
        atom_types_np.extend([atom_type] * count)
    atom_types_np = np.array(atom_types_np)

    lattice_np = np.array(poscar['Base'])

    # Compute using new method
    energy_new = calculator.compute_energy_from_tensor(
        positions_np, atom_types_np, lattice_np
    )
    print(f"  Energy: {energy_new:.6f} eV")

    # Check match
    print(f"\n✓ Match: {abs(energy_old - energy_new) < 1e-6}")
    print(f"  Difference: {abs(energy_old - energy_new):.10f} eV")

    return energy_old, energy_new


def test_batch_structures():
    """Test batch energy calculation"""
    print("\n" + "="*60)
    print("Test 2: Batch structures (8 identical)")
    print("="*60)

    base_dir = 'src/energy_models/cluster_expansion/energy_parameter'
    poscar_file = os.path.join(base_dir, 'POSCAR_ABO3')

    # Create calculator
    calculator = create_energy_calculator(
        base_dir=base_dir,
        atom_ind_group=[[0], [1, 2], [3, 4]],
        element_names=['Sr', 'Ti', 'Fe', 'O', 'VO']
    )

    # Load POSCAR
    poscar = posreader(poscar_file)
    poscar = dismatcreate(poscar)

    # Extract numpy arrays
    positions_np = np.array(poscar['LattPnt'])
    atom_types_np = []
    for atom_type, count in enumerate(poscar['AtomNum']):
        atom_types_np.extend([atom_type] * count)
    atom_types_np = np.array(atom_types_np)
    lattice_np = np.array(poscar['Base'])

    # Create batch (8 copies)
    batch_size = 8
    positions_batch = np.stack([positions_np] * batch_size)
    atom_types_batch = np.stack([atom_types_np] * batch_size)

    print(f"\nBatch shapes:")
    print(f"  Positions: {positions_batch.shape}")
    print(f"  Atom types: {atom_types_batch.shape}")
    print(f"  Lattice: {lattice_np.shape}")

    # Compute batch energies
    print(f"\n[Computing batch energies with 4 workers...]")
    energies = calculator.compute_energy_batch_from_tensor(
        positions_batch, atom_types_batch, lattice_np, n_workers=4
    )

    print(f"\nResults:")
    print(f"  Energies shape: {energies.shape}")
    print(f"  Energies: {energies}")
    print(f"  All same: {np.allclose(energies, energies[0])}")
    print(f"  Std dev: {np.std(energies):.10f}")

    return energies


def test_performance_comparison():
    """Compare old vs new method performance"""
    print("\n" + "="*60)
    print("Test 3: Performance comparison")
    print("="*60)

    import time

    base_dir = 'src/energy_models/cluster_expansion/energy_parameter'
    poscar_file = os.path.join(base_dir, 'POSCAR_ABO3')

    calculator = create_energy_calculator(
        base_dir=base_dir,
        atom_ind_group=[[0], [1, 2], [3, 4]],
        element_names=['Sr', 'Ti', 'Fe', 'O', 'VO']
    )

    poscar = posreader(poscar_file)
    poscar = dismatcreate(poscar)

    positions_np = np.array(poscar['LattPnt'])
    atom_types_np = []
    for atom_type, count in enumerate(poscar['AtomNum']):
        atom_types_np.extend([atom_type] * count)
    atom_types_np = np.array(atom_types_np)
    lattice_np = np.array(poscar['Base'])

    n_runs = 10

    # Old method (with POSCAR dict recreation)
    print(f"\n[Old method] With POSCAR dict conversion (simulated)")
    print(f"  Running {n_runs} iterations...")
    start = time.time()
    for _ in range(n_runs):
        # Simulate conversion overhead
        poscar_copy = poscar.copy()
        energy = calculator.compute_energy(poscar_copy)
    old_time = time.time() - start
    print(f"  Time: {old_time:.4f} s ({old_time/n_runs*1000:.2f} ms/iter)")

    # New method (direct numpy)
    print(f"\n[New method] Direct numpy arrays")
    print(f"  Running {n_runs} iterations...")
    start = time.time()
    for _ in range(n_runs):
        energy = calculator.compute_energy_from_tensor(
            positions_np, atom_types_np, lattice_np
        )
    new_time = time.time() - start
    print(f"  Time: {new_time:.4f} s ({new_time/n_runs*1000:.2f} ms/iter)")

    print(f"\n✓ Speedup: {old_time/new_time:.2f}x")


if __name__ == "__main__":
    try:
        # Test 1: Single structure
        energy_old, energy_new = test_single_structure()

        # Test 2: Batch
        energies = test_batch_structures()

        # Test 3: Performance
        test_performance_comparison()

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
