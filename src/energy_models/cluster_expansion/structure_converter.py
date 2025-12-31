"""
POSCAR <-> Tensor Converter for Diffusion Model

Converts between VASP POSCAR format and PyTorch tensors for diffusion model.
"""

import torch
import numpy as np
import copy
from src.energy_models.cluster_expansion.structure_utils import (
    posreader, poswriter, dismatcreate
)


class StructureConverter:
    """
    Convert between POSCAR dict and PyTorch tensors.

    Tensor format:
    - positions: [N, 3] fractional coordinates
    - atom_types: [N] integer type indices
    - lattice: [3, 3] lattice vectors

    POSCAR format:
    - Standard VASP POSCAR dictionary
    """

    def __init__(self, element_names=None, template_file=None):
        """
        Initialize converter.

        Args:
            element_names: List of element names (e.g., ['Sr', 'Ti', 'Fe', 'O', 'VO'])
            template_file: Optional template POSCAR for lattice/metadata
        """
        self.element_names = element_names

        if template_file:
            self.template = posreader(template_file)
        else:
            self.template = None


    def poscar_to_tensor(self, poscar, device='cpu'):
        """
        Convert POSCAR dict to PyTorch tensors.

        Args:
            poscar: POSCAR dictionary from posreader()
            device: PyTorch device ('cpu' or 'cuda')

        Returns:
            dict with keys:
                - positions: [N, 3] fractional coordinates (torch.Tensor)
                - atom_types: [N] type indices (torch.LongTensor)
                - lattice: [3, 3] lattice vectors (torch.Tensor)
                - metadata: dict with element names, counts, etc.
        """
        # Positions (fractional coordinates)
        positions = torch.tensor(poscar['LattPnt'], dtype=torch.float32, device=device)

        # Atom types (expanded from AtomNum)
        atom_types_list = []
        for type_idx, count in enumerate(poscar['AtomNum']):
            atom_types_list.extend([type_idx] * count)
        atom_types = torch.tensor(atom_types_list, dtype=torch.long, device=device)

        # Lattice vectors
        lattice = torch.tensor(poscar['Base'], dtype=torch.float32, device=device)

        # Metadata
        metadata = {
            'element_names': poscar['EleName'],
            'atom_counts': poscar['AtomNum'],
            'total_atoms': poscar['AtomSum'],
            'lattice_constant': poscar['LattConst'],
            'cell_name': poscar['CellName']
        }

        return {
            'positions': positions,
            'atom_types': atom_types,
            'lattice': lattice,
            'metadata': metadata
        }


    def tensor_to_poscar(self, positions, atom_types, lattice=None, metadata=None):
        """
        Convert tensors back to POSCAR dict.

        Args:
            positions: [N, 3] tensor of fractional coordinates
            atom_types: [N] tensor of type indices
            lattice: Optional [3, 3] tensor of lattice vectors
            metadata: Optional dict with element names, etc.

        Returns:
            poscar: POSCAR dictionary compatible with poswriter()
        """
        # Convert to numpy
        if torch.is_tensor(positions):
            positions = positions.cpu().numpy()
        if torch.is_tensor(atom_types):
            atom_types = atom_types.cpu().numpy()
        if lattice is not None and torch.is_tensor(lattice):
            lattice = lattice.cpu().numpy()

        # Create POSCAR dict
        poscar = {}

        # Use template if available
        if self.template:
            poscar = copy.deepcopy(self.template)
        else:
            poscar['CellName'] = metadata.get('cell_name', 'Structure') if metadata else 'Structure'
            poscar['LattConst'] = metadata.get('lattice_constant', 1.0) if metadata else 1.0
            poscar['IsSel'] = 0
            poscar['LatType'] = 'Direct'

        # Lattice
        if lattice is not None:
            poscar['Base'] = lattice.tolist()
        elif 'Base' not in poscar:
            raise ValueError("Lattice must be provided or template must be set")

        # Element names
        if metadata and 'element_names' in metadata:
            element_names = metadata['element_names']
        elif self.element_names:
            element_names = self.element_names
        else:
            # Auto-generate
            unique_types = sorted(set(atom_types.tolist()))
            element_names = [f'Elem{t}' for t in unique_types]

        poscar['EleName'] = element_names
        poscar['EleNum'] = len(element_names)

        # Count atoms per type
        unique_types = sorted(set(atom_types.tolist()))
        atom_counts = [np.sum(atom_types == t) for t in unique_types]
        poscar['AtomNum'] = atom_counts
        poscar['AtomSum'] = int(np.sum(atom_counts))

        # Positions
        poscar['LattPnt'] = positions.tolist()

        return poscar


    def batch_poscar_to_tensor(self, poscar_list, device='cpu'):
        """
        Convert batch of POSCAR dicts to batched tensors.

        Args:
            poscar_list: List of POSCAR dicts
            device: PyTorch device

        Returns:
            dict with:
                - positions: [B, N, 3]
                - atom_types: [B, N]
                - lattice: [B, 3, 3] or [3, 3] if same for all
                - metadata: list of metadata dicts
        """
        batch_data = [self.poscar_to_tensor(p, device=device) for p in poscar_list]

        # Stack tensors
        positions = torch.stack([d['positions'] for d in batch_data], dim=0)
        atom_types = torch.stack([d['atom_types'] for d in batch_data], dim=0)

        # Check if all lattices are the same
        lattices = [d['lattice'] for d in batch_data]
        if all(torch.allclose(lattices[0], l) for l in lattices):
            lattice = lattices[0]  # [3, 3]
        else:
            lattice = torch.stack(lattices, dim=0)  # [B, 3, 3]

        metadata = [d['metadata'] for d in batch_data]

        return {
            'positions': positions,
            'atom_types': atom_types,
            'lattice': lattice,
            'metadata': metadata
        }


    def batch_tensor_to_poscar(self, positions, atom_types, lattice=None, metadata=None):
        """
        Convert batched tensors to list of POSCAR dicts.

        Args:
            positions: [B, N, 3]
            atom_types: [B, N]
            lattice: [3, 3] or [B, 3, 3]
            metadata: Optional list of metadata dicts

        Returns:
            poscar_list: List of POSCAR dicts
        """
        batch_size = positions.shape[0]
        poscar_list = []

        for i in range(batch_size):
            pos_i = positions[i]
            types_i = atom_types[i]

            # Handle lattice
            if lattice.dim() == 2:
                lat_i = lattice
            else:
                lat_i = lattice[i]

            # Handle metadata
            meta_i = metadata[i] if metadata else None

            poscar_i = self.tensor_to_poscar(pos_i, types_i, lat_i, meta_i)
            poscar_list.append(poscar_i)

        return poscar_list


def fractional_to_cartesian(positions, lattice):
    """
    Convert fractional to Cartesian coordinates.

    Args:
        positions: [..., N, 3] fractional coordinates
        lattice: [3, 3] or [..., 3, 3] lattice vectors

    Returns:
        cartesian: [..., N, 3] Cartesian coordinates
    """
    return torch.matmul(positions, lattice)


def cartesian_to_fractional(cartesian, lattice):
    """
    Convert Cartesian to fractional coordinates.

    Args:
        cartesian: [..., N, 3] Cartesian coordinates
        lattice: [3, 3] or [..., 3, 3] lattice vectors

    Returns:
        positions: [..., N, 3] fractional coordinates
    """
    lattice_inv = torch.inverse(lattice)
    return torch.matmul(cartesian, lattice_inv)


def apply_pbc(positions):
    """
    Apply periodic boundary conditions (wrap to [0, 1)).

    Args:
        positions: [..., N, 3] fractional coordinates

    Returns:
        wrapped: [..., N, 3] wrapped coordinates
    """
    return positions - torch.floor(positions)


def compute_distance_matrix_pbc(positions, lattice):
    """
    Compute distance matrix with periodic boundary conditions.

    Args:
        positions: [N, 3] fractional coordinates
        lattice: [3, 3] lattice vectors

    Returns:
        distances: [N, N] distance matrix in Angstroms
    """
    N = positions.shape[0]

    # Pairwise differences in fractional coordinates
    delta = positions.unsqueeze(0) - positions.unsqueeze(1)  # [N, N, 3]

    # Apply minimum image convention
    delta = torch.where(delta > 0.5, delta - 1.0, delta)
    delta = torch.where(delta <= -0.5, delta + 1.0, delta)

    # Convert to Cartesian
    cart_delta = torch.matmul(delta, lattice)  # [N, N, 3]

    # Compute distances
    distances = torch.norm(cart_delta, dim=-1)  # [N, N]

    return distances


# Example usage
if __name__ == "__main__":
    # Create converter
    converter = StructureConverter(
        element_names=['Sr', 'Ti', 'O'],
        template_file='src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3'
    )

    # Load POSCAR
    poscar = posreader('src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3')
    poscar = dismatcreate(poscar)

    print("Original POSCAR:")
    print(f"  Elements: {poscar['EleName']}")
    print(f"  Counts: {poscar['AtomNum']}")
    print(f"  Total: {poscar['AtomSum']}")

    # Convert to tensor
    tensor_data = converter.poscar_to_tensor(poscar, device='cpu')

    print("\nTensor representation:")
    print(f"  Positions shape: {tensor_data['positions'].shape}")
    print(f"  Atom types shape: {tensor_data['atom_types'].shape}")
    print(f"  Lattice shape: {tensor_data['lattice'].shape}")
    print(f"  First 5 positions:\n{tensor_data['positions'][:5]}")
    print(f"  First 5 atom types: {tensor_data['atom_types'][:5]}")

    # Convert back
    poscar_restored = converter.tensor_to_poscar(
        tensor_data['positions'],
        tensor_data['atom_types'],
        tensor_data['lattice'],
        tensor_data['metadata']
    )

    print("\nRestored POSCAR:")
    print(f"  Elements: {poscar_restored['EleName']}")
    print(f"  Counts: {poscar_restored['AtomNum']}")
    print(f"  Total: {poscar_restored['AtomSum']}")

    # Test batch conversion
    poscar_list = [poscar, poscar, poscar]
    batch_data = converter.batch_poscar_to_tensor(poscar_list, device='cpu')

    print("\nBatch tensor representation:")
    print(f"  Positions shape: {batch_data['positions'].shape}")
    print(f"  Atom types shape: {batch_data['atom_types'].shape}")
    print(f"  Lattice shape: {batch_data['lattice'].shape}")

    # Test distance matrix
    distances = compute_distance_matrix_pbc(
        tensor_data['positions'],
        tensor_data['lattice']
    )
    print(f"\nDistance matrix shape: {distances.shape}")
    print(f"  Min distance (non-zero): {distances[distances > 0].min():.3f} Å")
    print(f"  Max distance: {distances.max():.3f} Å")
