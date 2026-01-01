"""
POSCAR <-> Tensor Conversion Example

Demonstrates how to use tensor interface with EnergyCalculator
for diffusion model integration.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.energy_models.cluster_expansion.energy_calculator import create_energy_calculator
from src.energy_models.cluster_expansion.structure_converter import StructureConverter
from src.energy_models.cluster_expansion.structure_utils import posreader, dismatcreate


def example_1_basic_tensor_conversion():
    """Example 1: Basic POSCAR to Tensor conversion"""
    print("=" * 60)
    print("Example 1: POSCAR → Tensor Conversion")
    print("=" * 60)

    # Load POSCAR
    base_dir = 'src/energy_models/cluster_expansion/energy_parameter'
    poscar_file = os.path.join(base_dir, 'POSCAR_ABO3')

    poscar = posreader(poscar_file)
    poscar = dismatcreate(poscar)

    print(f"\nOriginal POSCAR:")
    print(f"  Elements: {poscar['EleName']}")
    print(f"  Counts: {poscar['AtomNum']}")
    print(f"  Total atoms: {poscar['AtomSum']}")

    # Create converter
    converter = StructureConverter(
        element_names=['Sr', 'Ti', 'Fe', 'O', 'VO'],
        template_file=poscar_file
    )

    # Convert to tensor
    tensor_data = converter.poscar_to_tensor(poscar, device='cpu')

    print(f"\nTensor representation:")
    print(f"  Positions: {tensor_data['positions'].shape} {tensor_data['positions'].dtype}")
    print(f"  Atom types: {tensor_data['atom_types'].shape} {tensor_data['atom_types'].dtype}")
    print(f"  Lattice: {tensor_data['lattice'].shape} {tensor_data['lattice'].dtype}")

    print(f"\n  First 5 positions:")
    print(tensor_data['positions'][:5])

    print(f"\n  Atom type distribution:")
    element_names = ['Sr', 'Ti', 'Fe', 'O', 'VO']
    for i in range(5):
        count = (tensor_data['atom_types'] == i).sum().item()
        print(f"    Type {i} ({element_names[i]}): {count} atoms")

    # Convert back
    poscar_restored = converter.tensor_to_poscar(
        tensor_data['positions'],
        tensor_data['atom_types'],
        tensor_data['lattice'],
        tensor_data['metadata']
    )

    print(f"\nRestored POSCAR:")
    print(f"  Elements: {poscar_restored['EleName']}")
    print(f"  Counts: {poscar_restored['AtomNum']}")
    print(f"  Match: {poscar_restored['AtomNum'] == poscar['AtomNum']}")


def example_2_energy_from_tensor():
    """Example 2: Compute energy from tensors"""
    print("\n" + "=" * 60)
    print("Example 2: Energy Calculation from Tensors")
    print("=" * 60)

    base_dir = 'src/energy_models/cluster_expansion/energy_parameter'
    poscar_file = os.path.join(base_dir, 'POSCAR_ABO3')

    # Create calculator with tensor support
    calculator = create_energy_calculator(
        base_dir=base_dir,
        atom_ind_group=[[0], [1, 2], [3, 4]],  # Sr, Ti/Fe, O/VO
        element_names=['Sr', 'Ti', 'Fe', 'O', 'VO']
    )

    # Load and convert to tensor
    poscar = posreader(poscar_file)
    poscar = dismatcreate(poscar)

    tensor_data = calculator.converter.poscar_to_tensor(poscar, device='cpu')

    # Method 1: Compute from POSCAR
    print("\n[Method 1] Energy from POSCAR:")
    energy_poscar = calculator.compute_energy(poscar)
    print(f"  Energy: {energy_poscar:.6f} eV")

    # Method 2: Compute from tensors
    print("\n[Method 2] Energy from tensors:")
    energy_tensor = calculator.compute_energy_from_tensor(
        tensor_data['positions'],
        tensor_data['atom_types'],
        tensor_data['lattice'],
        tensor_data['metadata']
    )
    print(f"  Energy: {energy_tensor:.6f} eV")

    # Verify they match
    print(f"\n  Match: {abs(energy_poscar - energy_tensor) < 1e-6}")


def example_3_batch_tensor():
    """Example 3: Batch processing with tensors"""
    print("\n" + "=" * 60)
    print("Example 3: Batch Energy from Tensors")
    print("=" * 60)

    base_dir = 'src/energy_models/cluster_expansion/energy_parameter'
    poscar_file = os.path.join(base_dir, 'POSCAR_ABO3')

    # Create calculator
    calculator = create_energy_calculator(
        base_dir=base_dir,
        atom_ind_group=[[0], [1, 2], [3, 4]],
        element_names=['Sr', 'Ti', 'Fe', 'O', 'VO']
    )

    # Create batch of structures (copy same structure 8 times)
    poscar = posreader(poscar_file)
    poscar = dismatcreate(poscar)

    poscar_list = [poscar] * 8

    # Convert to batched tensors
    batch_data = calculator.converter.batch_poscar_to_tensor(poscar_list, device='cpu')

    print(f"\nBatch tensors:")
    print(f"  Positions: {batch_data['positions'].shape}")
    print(f"  Atom types: {batch_data['atom_types'].shape}")
    print(f"  Lattice: {batch_data['lattice'].shape}")

    # Compute energies
    print(f"\n[Computing batch energies with 4 workers...]")
    energies = calculator.compute_energy_batch_from_tensor(
        batch_data['positions'],
        batch_data['atom_types'],
        batch_data['lattice'],
        batch_data['metadata'],
        n_workers=4
    )

    print(f"\nResults:")
    print(f"  Energies shape: {energies.shape}")
    print(f"  Energies: {energies}")
    print(f"  All same (expected): {torch.allclose(torch.tensor(energies), torch.tensor(energies[0]).expand(8))}")


def example_4_diffusion_integration():
    """Example 4: Diffusion model integration pattern"""
    print("\n" + "=" * 60)
    print("Example 4: Diffusion Model Integration Pattern")
    print("=" * 60)

    base_dir = 'src/energy_models/cluster_expansion/energy_parameter'

    # Initialize calculator (do this once)
    calculator = create_energy_calculator(
        base_dir=base_dir,
        atom_ind_group=[[0], [1, 2], [3, 4]],
        element_names=['Sr', 'Ti', 'Fe', 'O', 'VO']
    )

    print("\n[Simulating diffusion model workflow...]")

    # Simulate diffusion model generating structures
    print("\n1. Diffusion model generates batch of structures (GPU)")
    # In real case: x_generated = diffusion_model(noise)
    # Here we just load template
    poscar_file = os.path.join(base_dir, 'POSCAR_ABO3')
    poscar = posreader(poscar_file)
    poscar = dismatcreate(poscar)

    # Batch of 16 structures
    batch_size = 16
    poscar_list = [poscar] * batch_size

    # Convert to tensors (what diffusion model would output)
    batch_data = calculator.converter.batch_poscar_to_tensor(poscar_list)
    positions = batch_data['positions']  # [16, 160, 3]
    atom_types = batch_data['atom_types']  # [16, 160]
    lattice = batch_data['lattice']  # [3, 3]

    print(f"   Generated structures: {positions.shape}")

    # Transfer to CPU for energy calculation (if on GPU)
    print("\n2. Transfer to CPU for energy oracle")
    positions_cpu = positions.cpu()
    atom_types_cpu = atom_types.cpu()
    lattice_cpu = lattice.cpu()

    # Compute energies in parallel
    print("\n3. Compute energies in parallel (CPU, 8 workers)")
    energies = calculator.compute_energy_batch_from_tensor(
        positions_cpu,
        atom_types_cpu,
        lattice_cpu,
        n_workers=8
    )

    print(f"   Energies computed: {energies.shape}")
    print(f"   Energy range: [{energies.min():.4f}, {energies.max():.4f}]")

    # Convert back to tensor for loss computation
    print("\n4. Convert to PyTorch tensor for loss")
    energies_tensor = torch.tensor(energies, dtype=torch.float32)

    # Simulated loss computation
    target_energy = -10.0  # Example target
    loss = torch.mean((energies_tensor - target_energy) ** 2)

    print(f"   Energies (tensor): {energies_tensor.shape} {energies_tensor.dtype}")
    print(f"   Example loss: {loss.item():.6f}")

    print("\n5. Backpropagation through diffusion model (not through energy)")
    print("   loss.backward() - gradients only for diffusion parameters")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("POSCAR <-> Tensor Conversion Examples")
    print("="*60)

    try:
        example_1_basic_tensor_conversion()
        example_2_energy_from_tensor()
        example_3_batch_tensor()
        example_4_diffusion_integration()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
