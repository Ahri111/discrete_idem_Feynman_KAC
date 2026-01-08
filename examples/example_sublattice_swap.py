"""
Sublattice-Constrained Swap Diffusion (PyTorch GPU)

Fully parallelized batch processing for ABO3 perovskite structures.
- B-site: Ti ↔ Fe only
- O-site: O ↔ VO only
- A-site (Sr): fixed
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# POSCAR Parser
# =============================================================================

def parse_poscar_string(poscar_str: str) -> dict:
    """Parse POSCAR string to dict"""
    lines = [l.strip() for l in poscar_str.strip().split('\n')]

    cell_name = lines[0]
    latt_const = float(lines[1])
    base = [[float(x) for x in lines[i].split()] for i in range(2, 5)]
    ele_names = lines[5].split()
    atom_nums = [int(x) for x in lines[6].split()]

    positions = []
    for i in range(8, 8 + sum(atom_nums)):
        parts = lines[i].split()
        positions.append([float(parts[0]), float(parts[1]), float(parts[2])])

    return {
        'CellName': cell_name,
        'LattConst': latt_const,
        'Base': base,
        'EleName': ele_names,
        'AtomNum': atom_nums,
        'AtomSum': sum(atom_nums),
        'LattPnt': positions
    }


def poscar_to_tensors(poscar: dict, device='cpu') -> dict:
    """Convert POSCAR to tensors with sublattice masks"""

    # Positions [N, 3]
    positions = torch.tensor(poscar['LattPnt'], dtype=torch.float32, device=device)

    # Atom types [N]
    atom_types = []
    for type_idx, count in enumerate(poscar['AtomNum']):
        atom_types.extend([type_idx] * count)
    atom_types = torch.tensor(atom_types, dtype=torch.long, device=device)

    # Lattice [3, 3]
    lattice = torch.tensor(poscar['Base'], dtype=torch.float32, device=device)

    # Type mapping: Sr=0, Ti=1, Fe=2, O=3, VO=4
    ele_names = poscar['EleName']
    type_map = {name: idx for idx, name in enumerate(ele_names)}

    # Sublattice masks
    b_site_mask = (atom_types == type_map['Ti']) | (atom_types == type_map['Fe'])
    o_site_mask = (atom_types == type_map['O']) | (atom_types == type_map['VO'])

    return {
        'positions': positions,
        'atom_types': atom_types,
        'lattice': lattice,
        'b_site_mask': b_site_mask,
        'o_site_mask': o_site_mask,
        'element_names': ele_names,
        'type_map': type_map,
        'atom_counts': poscar['AtomNum']
    }


# =============================================================================
# Parallel Swap Operations (from symdiff/utils.py pattern)
# =============================================================================

@torch.no_grad()
def swap_by_idx(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Swap elements at idx positions (fully parallel)

    Args:
        x: [batch, N] atom types
        idx: [batch, 2] indices to swap

    Returns:
        x_swapped: [batch, N]
    """
    first = x.gather(-1, idx[..., 0:1])   # [batch, 1]
    second = x.gather(-1, idx[..., 1:2])  # [batch, 1]

    x_swapped = x.clone()
    x_swapped.scatter_(-1, idx[..., 0:1], second)
    x_swapped.scatter_(-1, idx[..., 1:2], first)

    return x_swapped


@torch.no_grad()
def sample_sublattice_swap(
    atom_types: torch.Tensor,       # [batch, N]
    sublattice_mask: torch.Tensor,  # [N]
    type_a: int,
    type_b: int,
    scores: torch.Tensor = None,    # [batch, N] or None
) -> tuple:
    """
    GPU-parallel sublattice-constrained swap

    Selects one atom of type_a and one of type_b within sublattice,
    then swaps them. All batch elements processed in parallel.

    Args:
        atom_types: [batch, N] current atom type indices
        sublattice_mask: [N] boolean mask for sublattice
        type_a, type_b: types to swap (e.g., Ti=1, Fe=2)
        scores: [batch, N] swap scores (None = uniform random)

    Returns:
        swapped: [batch, N] atom types after swap
        indices: [batch, 2] swapped positions
    """
    device = atom_types.device
    batch_size, N = atom_types.shape

    # 1. Get sublattice indices
    sub_idx = torch.where(sublattice_mask)[0]  # [M]
    M = len(sub_idx)

    # 2. Extract sublattice
    sub_types = atom_types[:, sub_idx]  # [batch, M]

    # 3. Type masks
    is_a = (sub_types == type_a)  # [batch, M]
    is_b = (sub_types == type_b)  # [batch, M]

    # 4. Scores (uniform if not provided)
    if scores is None:
        sub_scores = torch.zeros(batch_size, M, device=device)
    else:
        sub_scores = scores[:, sub_idx]

    # 5. Gumbel noise for stochastic sampling
    noise = torch.rand(batch_size, M, device=device).clamp(min=1e-10)
    gumbel = -torch.log(-torch.log(noise))

    # 6. Masked scores (type_a positions only, type_b positions only)
    score_a = sub_scores + gumbel
    score_b = sub_scores + gumbel
    score_a = score_a.masked_fill(~is_a, float('-inf'))
    score_b = score_b.masked_fill(~is_b, float('-inf'))

    # 7. Select one from each type (parallel across batch)
    local_a = torch.argmax(score_a, dim=-1)  # [batch]
    local_b = torch.argmax(score_b, dim=-1)  # [batch]

    # 8. Map to global indices
    global_a = sub_idx[local_a]  # [batch]
    global_b = sub_idx[local_b]  # [batch]

    # 9. Stack indices and swap
    indices = torch.stack([global_a, global_b], dim=-1)  # [batch, 2]
    swapped = swap_by_idx(atom_types, indices)

    return swapped, indices


@torch.no_grad()
def apply_n_swaps(
    atom_types: torch.Tensor,       # [batch, N]
    b_site_mask: torch.Tensor,      # [N]
    o_site_mask: torch.Tensor,      # [N]
    type_map: dict,
    n_swaps: int,
    swap_mode: str = 'both'
) -> tuple:
    """
    Apply n swap steps (batch parallel at each step)

    Args:
        atom_types: [batch, N]
        b_site_mask, o_site_mask: sublattice masks
        type_map: element name -> type index
        n_swaps: number of swap steps
        swap_mode: 'B-site', 'O-site', or 'both'

    Returns:
        final: [batch, N] after all swaps
        history: list of (sublattice, indices) tuples
    """
    current = atom_types.clone()
    history = []

    ti, fe = type_map['Ti'], type_map['Fe']
    o, vo = type_map['O'], type_map['VO']

    for step in range(n_swaps):
        # Choose sublattice
        if swap_mode == 'B-site':
            do_b = True
        elif swap_mode == 'O-site':
            do_b = False
        else:
            do_b = torch.rand(1).item() < 0.5

        # Apply swap (entire batch in parallel)
        if do_b:
            current, idx = sample_sublattice_swap(current, b_site_mask, ti, fe)
            history.append(('B', idx.clone()))
        else:
            current, idx = sample_sublattice_swap(current, o_site_mask, o, vo)
            history.append(('O', idx.clone()))

    return current, history


# =============================================================================
# Visualization
# =============================================================================

def print_composition(atom_types: torch.Tensor, element_names: list):
    """Print composition for all batches"""
    batch_size = atom_types.shape[0]

    print(f"\n{'Batch':>5} | " + " | ".join(f"{n:>3}" for n in element_names))
    print("-" * 50)

    for b in range(batch_size):
        counts = [(atom_types[b] == t).sum().item() for t in range(len(element_names))]
        print(f"{b:>5} | " + " | ".join(f"{c:>3}" for c in counts))


def print_sublattice(atom_types: torch.Tensor, mask: torch.Tensor, type_map: dict,
                     name: str, type_names: tuple, batch_idx: int = 0):
    """Print sublattice configuration"""
    sub_idx = torch.where(mask)[0]
    sub_types = atom_types[batch_idx, sub_idx].cpu()

    symbols = [type_names[0] if t == type_map[type_names[0]] else type_names[1]
               for t in sub_types.tolist()]

    print(f"\n  {name} ({len(sub_idx)} sites):")
    for i in range(0, len(symbols), 8):
        row = symbols[i:i+8]
        print(f"    {i:>2}-{i+len(row)-1:>2}: " + " ".join(f"{s:>2}" for s in row))


# =============================================================================
# Main
# =============================================================================

POSCAR_STR = """SrTiFeO
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


def main():
    torch.manual_seed(42)

    print("=" * 60)
    print("Sublattice-Constrained Swap Diffusion (PyTorch)")
    print("=" * 60)

    # 1. Parse & convert
    print("\n[1] Parsing POSCAR...")
    poscar = parse_poscar_string(POSCAR_STR)
    print(f"    Elements: {poscar['EleName']}")
    print(f"    Counts:   {poscar['AtomNum']}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[2] Converting to tensors (device: {device})...")
    data = poscar_to_tensors(poscar, device=device)
    print(f"    Shape: {data['atom_types'].shape}")
    print(f"    B-site: {data['b_site_mask'].sum().item()} atoms (Ti+Fe)")
    print(f"    O-site: {data['o_site_mask'].sum().item()} atoms (O+VO)")

    # 2. Create batch
    batch_size = 16
    print(f"\n[3] Creating batch of {batch_size}...")
    x = data['atom_types'].unsqueeze(0).expand(batch_size, -1).clone()
    print(f"    Batch shape: {x.shape}")

    # 3. Show original
    print("\n[4] Original B-site configuration (batch 0):")
    print_sublattice(x, data['b_site_mask'], data['type_map'],
                     "B-site", ('Ti', 'Fe'), batch_idx=0)

    # 4. Apply swaps (PARALLEL across batch)
    n_swaps = 20
    print(f"\n[5] Applying {n_swaps} swaps (batch-parallel)...")

    x_noisy, history = apply_n_swaps(
        x,
        data['b_site_mask'],
        data['o_site_mask'],
        data['type_map'],
        n_swaps=n_swaps,
        swap_mode='both'
    )

    # Show swap history (first 5 steps)
    print("\n    Swap history (first 5 steps):")
    for i, (site, idx) in enumerate(history[:5]):
        # idx is [batch, 2], show for batch 0
        a, b = idx[0].tolist()
        print(f"      Step {i+1}: {site}-site swap ({a}, {b})")
    print(f"      ... and {n_swaps - 5} more")

    # 5. Show results
    print("\n[6] After swaps - B-site configuration:")
    for b in [0, 7, 15]:
        print(f"\n    Batch {b}:")
        print_sublattice(x_noisy, data['b_site_mask'], data['type_map'],
                         "B-site", ('Ti', 'Fe'), batch_idx=b)

    # 6. Verify composition
    print("\n[7] Composition check:")
    print_composition(x_noisy, data['element_names'])

    original = torch.tensor(data['atom_counts'], device=device)
    all_ok = all(
        all((x_noisy[b] == t).sum() == original[t] for t in range(len(original)))
        for b in range(batch_size)
    )
    print(f"\n    Composition preserved: {'✓' if all_ok else '✗'}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
