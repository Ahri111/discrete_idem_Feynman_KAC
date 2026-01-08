"""
Sublattice-Constrained Swap Diffusion Example

Demonstrates:
1. Parse POSCAR with sublattice structure
2. Create batch of 16 structures
3. Apply constrained swaps (Ti↔Fe on B-site, O↔VO on O-site)
4. Visualize the swaps
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.energy_models.cluster_expansion.structure_utils import posreader


# =============================================================================
# POSCAR Parser for Sublattice Structure
# =============================================================================

def parse_poscar_string(poscar_str: str) -> dict:
    """Parse POSCAR string into structured dict with sublattice info"""
    lines = [l.strip() for l in poscar_str.strip().split('\n')]

    # Parse header
    cell_name = lines[0]
    latt_const = float(lines[1])

    # Lattice vectors
    base = []
    for i in range(2, 5):
        base.append([float(x) for x in lines[i].split()])

    # Element names and counts
    ele_names = lines[5].split()
    atom_nums = [int(x) for x in lines[6].split()]
    atom_sum = sum(atom_nums)

    # Coordinate type
    coord_type = lines[7]

    # Positions
    positions = []
    for i in range(8, 8 + atom_sum):
        parts = lines[i].split()
        pos = [float(parts[0]), float(parts[1]), float(parts[2])]
        positions.append(pos)

    return {
        'CellName': cell_name,
        'LattConst': latt_const,
        'Base': base,
        'EleName': ele_names,
        'AtomNum': atom_nums,
        'AtomSum': atom_sum,
        'LatType': coord_type,
        'LattPnt': positions
    }


def poscar_to_tensors(poscar: dict, device='cpu') -> dict:
    """Convert POSCAR dict to PyTorch tensors with sublattice masks"""

    # Positions [N, 3]
    positions = torch.tensor(poscar['LattPnt'], dtype=torch.float32, device=device)

    # Atom types [N] - expanded from counts
    atom_types = []
    for type_idx, count in enumerate(poscar['AtomNum']):
        atom_types.extend([type_idx] * count)
    atom_types = torch.tensor(atom_types, dtype=torch.long, device=device)

    # Lattice [3, 3]
    lattice = torch.tensor(poscar['Base'], dtype=torch.float32, device=device)

    # Build sublattice masks
    # Sr=0 (A-site), Ti=1, Fe=2 (B-site), O=3, VO=4 (O-site)
    ele_names = poscar['EleName']

    # Find type indices
    type_map = {name: idx for idx, name in enumerate(ele_names)}

    a_site_mask = (atom_types == type_map.get('Sr', -1))
    b_site_mask = (atom_types == type_map.get('Ti', -1)) | (atom_types == type_map.get('Fe', -1))
    o_site_mask = (atom_types == type_map.get('O', -1)) | (atom_types == type_map.get('VO', -1))

    return {
        'positions': positions,
        'atom_types': atom_types,
        'lattice': lattice,
        'a_site_mask': a_site_mask,
        'b_site_mask': b_site_mask,
        'o_site_mask': o_site_mask,
        'element_names': ele_names,
        'type_map': type_map,
        'atom_counts': poscar['AtomNum']
    }


# =============================================================================
# GPU-Optimized Sublattice Swap
# =============================================================================

@torch.no_grad()
def sample_sublattice_swap(
    atom_types: torch.Tensor,      # [batch, N]
    sublattice_mask: torch.Tensor, # [N] boolean
    type_a: int,
    type_b: int,
    scores: torch.Tensor = None,   # [batch, N] optional, uniform if None
    deterministic: bool = False
) -> tuple:
    """
    GPU-optimized sublattice-constrained swap

    Args:
        atom_types: [batch, N] atom type indices
        sublattice_mask: [N] mask for sublattice sites
        type_a, type_b: types to swap (e.g., Ti=1, Fe=2)
        scores: optional swap scores, uniform if None
        deterministic: if True, pick highest score pairs

    Returns:
        swapped_types: [batch, N]
        swap_indices: [batch, 2] - indices that were swapped
    """
    device = atom_types.device
    batch_size, N = atom_types.shape

    # Get sublattice indices
    sub_indices = torch.where(sublattice_mask)[0]  # [M]
    M = len(sub_indices)

    # Extract sublattice atom types
    sub_types = atom_types[:, sub_indices]  # [batch, M]

    # Find type_a and type_b positions within sublattice
    is_type_a = (sub_types == type_a)  # [batch, M]
    is_type_b = (sub_types == type_b)  # [batch, M]

    # Create scores if not provided (uniform)
    if scores is None:
        scores = torch.zeros(batch_size, M, device=device)
    else:
        scores = scores[:, sub_indices]  # [batch, M]

    # Mask scores for each type
    score_a = scores.masked_fill(~is_type_a, float('-inf'))
    score_b = scores.masked_fill(~is_type_b, float('-inf'))

    # Gumbel sampling
    if not deterministic:
        noise_a = torch.rand_like(score_a)
        noise_a = torch.clamp(noise_a, min=1e-10)
        gumbel_a = -torch.log(-torch.log(noise_a))

        noise_b = torch.rand_like(score_b)
        noise_b = torch.clamp(noise_b, min=1e-10)
        gumbel_b = -torch.log(-torch.log(noise_b))

        score_a = score_a + gumbel_a
        score_b = score_b + gumbel_b

    # Select one from each type
    local_idx_a = torch.argmax(score_a, dim=-1)  # [batch]
    local_idx_b = torch.argmax(score_b, dim=-1)  # [batch]

    # Map to global indices
    global_idx_a = sub_indices[local_idx_a]  # [batch]
    global_idx_b = sub_indices[local_idx_b]  # [batch]

    # Perform swap
    swapped_types = atom_types.clone()
    batch_idx = torch.arange(batch_size, device=device)
    swapped_types[batch_idx, global_idx_a] = type_b
    swapped_types[batch_idx, global_idx_b] = type_a

    swap_indices = torch.stack([global_idx_a, global_idx_b], dim=-1)  # [batch, 2]

    return swapped_types, swap_indices


def apply_random_swaps(
    atom_types: torch.Tensor,
    b_site_mask: torch.Tensor,
    o_site_mask: torch.Tensor,
    type_map: dict,
    n_swaps: int = 10,
    swap_mode: str = 'both'
) -> tuple:
    """
    Apply multiple random swaps to create noisy configurations

    Args:
        atom_types: [batch, N]
        b_site_mask, o_site_mask: sublattice masks
        type_map: element name to type index mapping
        n_swaps: number of swap steps
        swap_mode: 'B-site', 'O-site', or 'both'

    Returns:
        noisy_types: [batch, N]
        swap_history: list of swap indices
    """
    current = atom_types.clone()
    swap_history = []

    ti_type = type_map.get('Ti', 1)
    fe_type = type_map.get('Fe', 2)
    o_type = type_map.get('O', 3)
    vo_type = type_map.get('VO', 4)

    for step in range(n_swaps):
        # Choose sublattice
        if swap_mode == 'B-site':
            do_b_site = True
        elif swap_mode == 'O-site':
            do_b_site = False
        else:  # 'both'
            do_b_site = (torch.rand(1).item() < 0.5)

        if do_b_site:
            current, indices = sample_sublattice_swap(
                current, b_site_mask, ti_type, fe_type
            )
        else:
            current, indices = sample_sublattice_swap(
                current, o_site_mask, o_type, vo_type
            )

        swap_history.append(('B' if do_b_site else 'O', indices))

    return current, swap_history


# =============================================================================
# Visualization
# =============================================================================

def visualize_atom_types(atom_types: torch.Tensor, element_names: list, atom_counts: list):
    """Print atom type distribution for each batch"""
    batch_size = atom_types.shape[0]

    print("\n" + "="*70)
    print("Atom Type Distribution (after swaps)")
    print("="*70)

    for b in range(batch_size):
        types_b = atom_types[b].cpu().numpy()

        # Count each type
        counts = {}
        for t, name in enumerate(element_names):
            counts[name] = (types_b == t).sum()

        print(f"\nBatch {b:2d}: ", end="")
        for name in element_names:
            print(f"{name}={counts[name]:2d}  ", end="")

        # Check composition preserved
        original_counts = {name: c for name, c in zip(element_names, atom_counts)}
        if all(counts[n] == original_counts[n] for n in element_names):
            print("✓ composition preserved")
        else:
            print("✗ COMPOSITION ERROR!")


def visualize_swaps(swap_history: list, batch_idx: int = 0):
    """Show swap history for one batch element"""
    print(f"\n--- Swap History (batch {batch_idx}) ---")
    for i, (site, indices) in enumerate(swap_history):
        idx_a = indices[batch_idx, 0].item()
        idx_b = indices[batch_idx, 1].item()
        print(f"  Step {i+1}: {site}-site swap at indices ({idx_a}, {idx_b})")


def visualize_sublattice_state(
    atom_types: torch.Tensor,
    b_site_mask: torch.Tensor,
    o_site_mask: torch.Tensor,
    type_map: dict,
    batch_idx: int = 0
):
    """Show detailed sublattice configuration"""
    types = atom_types[batch_idx].cpu()

    # B-site configuration
    b_indices = torch.where(b_site_mask)[0]
    b_types = types[b_indices]
    ti_count = (b_types == type_map['Ti']).sum().item()
    fe_count = (b_types == type_map['Fe']).sum().item()

    # O-site configuration
    o_indices = torch.where(o_site_mask)[0]
    o_types = types[o_indices]
    o_count = (o_types == type_map['O']).sum().item()
    vo_count = (o_types == type_map['VO']).sum().item()

    print(f"\n--- Sublattice State (batch {batch_idx}) ---")
    print(f"  B-site ({len(b_indices)} sites): Ti={ti_count}, Fe={fe_count}")
    print(f"  O-site ({len(o_indices)} sites): O={o_count}, VO={vo_count}")

    # Show B-site pattern (first 16)
    print(f"\n  B-site pattern (first 16 sites):")
    print("  ", end="")
    for i, idx in enumerate(b_indices[:16]):
        t = types[idx].item()
        symbol = 'Ti' if t == type_map['Ti'] else 'Fe'
        print(f"{symbol:2s} ", end="")
        if (i + 1) % 8 == 0:
            print("\n  ", end="")


# =============================================================================
# Main Demo
# =============================================================================

def main():
    # Sample POSCAR from user input
    poscar_str = """SrTiFeO
1.000000
11.199000 0.000000 0.000000
0.000000 11.199000 0.000000
0.000000 0.000000 15.983000
Sr Ti Fe O VO
32 16 16 88 8
Direct
0.000000 0.250000 0.125000
0.000000 0.250000 0.625000
0.000000 0.750000 0.125000
0.000000 0.750000 0.625000
0.500000 0.250000 0.125000
0.500000 0.250000 0.625000
0.500000 0.750000 0.125000
0.500000 0.750000 0.625000
0.000000 0.250000 0.375000
0.000000 0.250000 0.875000
0.000000 0.750000 0.375000
0.000000 0.750000 0.875000
0.500000 0.250000 0.375000
0.500000 0.250000 0.875000
0.500000 0.750000 0.375000
0.500000 0.750000 0.875000
0.250000 0.000000 0.125000
0.250000 0.000000 0.625000
0.250000 0.500000 0.125000
0.250000 0.500000 0.625000
0.750000 0.000000 0.125000
0.750000 0.000000 0.625000
0.750000 0.500000 0.125000
0.750000 0.500000 0.625000
0.250000 0.000000 0.375000
0.250000 0.000000 0.875000
0.250000 0.500000 0.375000
0.250000 0.500000 0.875000
0.750000 0.000000 0.375000
0.750000 0.000000 0.875000
0.750000 0.500000 0.375000
0.750000 0.500000 0.875000
0.000000 0.000000 0.000000
0.250000 0.750000 0.500000
0.750000 0.250000 0.500000
0.500000 0.000000 0.000000
0.000000 0.500000 0.000000
0.500000 0.500000 0.500000
0.750000 0.250000 0.750000
0.000000 0.500000 0.500000
0.750000 0.750000 0.500000
0.250000 0.750000 0.750000
0.250000 0.250000 0.750000
0.750000 0.750000 0.750000
0.500000 0.500000 0.000000
0.500000 0.000000 0.500000
0.000000 0.000000 0.500000
0.250000 0.250000 0.500000
0.750000 0.250000 0.000000
0.250000 0.750000 0.250000
0.500000 0.500000 0.750000
0.750000 0.750000 0.000000
0.000000 0.500000 0.250000
0.500000 0.000000 0.750000
0.250000 0.250000 0.250000
0.000000 0.500000 0.750000
0.750000 0.250000 0.250000
0.250000 0.750000 0.000000
0.000000 0.000000 0.250000
0.000000 0.000000 0.750000
0.750000 0.750000 0.250000
0.250000 0.250000 0.000000
0.500000 0.000000 0.250000
0.500000 0.500000 0.250000
0.000000 0.000000 0.125000
0.000000 0.000000 0.625000
0.000000 0.500000 0.125000
0.000000 0.500000 0.625000
0.500000 0.000000 0.125000
0.500000 0.000000 0.625000
0.500000 0.500000 0.125000
0.500000 0.500000 0.625000
0.000000 0.000000 0.375000
0.000000 0.000000 0.875000
0.000000 0.500000 0.375000
0.000000 0.500000 0.875000
0.500000 0.000000 0.375000
0.500000 0.000000 0.875000
0.500000 0.500000 0.375000
0.500000 0.500000 0.875000
0.250000 0.250000 0.375000
0.250000 0.250000 0.875000
0.250000 0.750000 0.375000
0.250000 0.750000 0.875000
0.750000 0.250000 0.375000
0.750000 0.250000 0.875000
0.750000 0.750000 0.375000
0.750000 0.750000 0.875000
0.250000 0.250000 0.125000
0.250000 0.250000 0.625000
0.250000 0.750000 0.125000
0.250000 0.750000 0.625000
0.750000 0.250000 0.125000
0.750000 0.250000 0.625000
0.750000 0.750000 0.125000
0.750000 0.750000 0.625000
0.141000 0.391000 0.250000
0.391000 0.359000 0.250000
0.641000 0.391000 0.250000
0.891000 0.359000 0.250000
0.141000 0.891000 0.250000
0.391000 0.859000 0.250000
0.641000 0.891000 0.250000
0.891000 0.859000 0.250000
0.141000 0.109000 0.500000
0.391000 0.141000 0.500000
0.641000 0.109000 0.500000
0.891000 0.141000 0.500000
0.109000 0.359000 0.500000
0.359000 0.391000 0.500000
0.609000 0.359000 0.500000
0.859000 0.391000 0.500000
0.141000 0.609000 0.500000
0.391000 0.641000 0.500000
0.641000 0.609000 0.500000
0.891000 0.641000 0.500000
0.109000 0.859000 0.500000
0.359000 0.891000 0.500000
0.609000 0.859000 0.500000
0.859000 0.891000 0.500000
0.109000 0.141000 0.750000
0.359000 0.109000 0.750000
0.609000 0.141000 0.750000
0.859000 0.109000 0.750000
0.141000 0.391000 0.750000
0.391000 0.359000 0.750000
0.641000 0.391000 0.750000
0.891000 0.359000 0.750000
0.109000 0.641000 0.750000
0.359000 0.609000 0.750000
0.609000 0.641000 0.750000
0.859000 0.609000 0.750000
0.141000 0.891000 0.750000
0.391000 0.859000 0.750000
0.641000 0.891000 0.750000
0.891000 0.859000 0.750000
0.141000 0.109000 0.000000
0.391000 0.141000 0.000000
0.641000 0.109000 0.000000
0.891000 0.141000 0.000000
0.109000 0.359000 0.000000
0.359000 0.391000 0.000000
0.609000 0.359000 0.000000
0.859000 0.391000 0.000000
0.141000 0.609000 0.000000
0.391000 0.641000 0.000000
0.641000 0.609000 0.000000
0.891000 0.641000 0.000000
0.109000 0.859000 0.000000
0.359000 0.891000 0.000000
0.609000 0.859000 0.000000
0.859000 0.891000 0.000000
0.109000 0.141000 0.250000
0.359000 0.109000 0.250000
0.609000 0.141000 0.250000
0.859000 0.109000 0.250000
0.109000 0.641000 0.250000
0.359000 0.609000 0.250000
0.609000 0.641000 0.250000
0.859000 0.609000 0.250000"""

    print("="*70)
    print("Sublattice-Constrained Swap Diffusion Demo")
    print("="*70)

    # 1. Parse POSCAR
    print("\n[1] Parsing POSCAR...")
    poscar = parse_poscar_string(poscar_str)
    print(f"    Elements: {poscar['EleName']}")
    print(f"    Counts:   {poscar['AtomNum']}")
    print(f"    Total:    {poscar['AtomSum']} atoms")

    # 2. Convert to tensors
    print("\n[2] Converting to tensors...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Device: {device}")

    data = poscar_to_tensors(poscar, device=device)
    print(f"    Positions shape: {data['positions'].shape}")
    print(f"    Atom types shape: {data['atom_types'].shape}")
    print(f"    B-site atoms: {data['b_site_mask'].sum().item()}")
    print(f"    O-site atoms: {data['o_site_mask'].sum().item()}")

    # 3. Create batch of 16
    print("\n[3] Creating batch of 16 structures...")
    batch_size = 16
    atom_types_batch = data['atom_types'].unsqueeze(0).expand(batch_size, -1).clone()
    print(f"    Batch shape: {atom_types_batch.shape}")

    # 4. Apply random swaps (different amounts per batch)
    print("\n[4] Applying random swaps...")

    # Different noise levels for different batches
    noisy_batch = atom_types_batch.clone()
    all_swap_history = []

    for b in range(batch_size):
        n_swaps = (b + 1) * 2  # 2, 4, 6, ..., 32 swaps
        single = atom_types_batch[b:b+1]

        noisy, history = apply_random_swaps(
            single,
            data['b_site_mask'],
            data['o_site_mask'],
            data['type_map'],
            n_swaps=n_swaps,
            swap_mode='both'
        )
        noisy_batch[b] = noisy[0]
        all_swap_history.append((n_swaps, history))

    print(f"    Applied 2 to 32 swaps per structure")

    # 5. Visualize results
    visualize_atom_types(
        noisy_batch,
        data['element_names'],
        data['atom_counts']
    )

    # Show detailed view for a few examples
    for b in [0, 7, 15]:
        n_swaps, history = all_swap_history[b]
        print(f"\n{'='*70}")
        print(f"Detailed view: Batch {b} ({n_swaps} swaps)")
        print("="*70)
        visualize_sublattice_state(
            noisy_batch,
            data['b_site_mask'],
            data['o_site_mask'],
            data['type_map'],
            batch_idx=b
        )
        if n_swaps <= 10:
            visualize_swaps(history, batch_idx=0)

    # 6. Verify composition preservation
    print("\n" + "="*70)
    print("Composition Verification")
    print("="*70)

    original_counts = torch.tensor(data['atom_counts'], device=device)
    all_correct = True

    for b in range(batch_size):
        types_b = noisy_batch[b]
        counts_b = torch.tensor([
            (types_b == t).sum().item()
            for t in range(len(data['element_names']))
        ], device=device)

        if not torch.equal(counts_b, original_counts):
            print(f"  Batch {b}: FAILED - {counts_b.tolist()} vs {original_counts.tolist()}")
            all_correct = False

    if all_correct:
        print("  ✓ All 16 structures have correct composition!")

    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    main()
