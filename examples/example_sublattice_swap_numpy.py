"""
Sublattice-Constrained Swap Diffusion Demo (NumPy version)

Demonstrates constrained swaps without PyTorch dependency.
"""

import numpy as np

# =============================================================================
# POSCAR Parser
# =============================================================================

def parse_poscar_string(poscar_str: str) -> dict:
    """Parse POSCAR string into structured dict"""
    lines = [l.strip() for l in poscar_str.strip().split('\n')]

    cell_name = lines[0]
    latt_const = float(lines[1])

    base = []
    for i in range(2, 5):
        base.append([float(x) for x in lines[i].split()])

    ele_names = lines[5].split()
    atom_nums = [int(x) for x in lines[6].split()]
    atom_sum = sum(atom_nums)

    positions = []
    for i in range(8, 8 + atom_sum):
        parts = lines[i].split()
        pos = [float(parts[0]), float(parts[1]), float(parts[2])]
        positions.append(pos)

    return {
        'CellName': cell_name,
        'EleName': ele_names,
        'AtomNum': atom_nums,
        'AtomSum': atom_sum,
        'LattPnt': np.array(positions)
    }


def poscar_to_arrays(poscar: dict) -> dict:
    """Convert POSCAR to numpy arrays with sublattice info"""

    # Atom types [N] - expanded from counts
    atom_types = []
    for type_idx, count in enumerate(poscar['AtomNum']):
        atom_types.extend([type_idx] * count)
    atom_types = np.array(atom_types, dtype=np.int32)

    # Type mapping
    ele_names = poscar['EleName']
    type_map = {name: idx for idx, name in enumerate(ele_names)}

    # Sublattice masks
    # Sr=0 (A-site), Ti=1, Fe=2 (B-site), O=3, VO=4 (O-site)
    b_site_mask = (atom_types == type_map.get('Ti', -1)) | (atom_types == type_map.get('Fe', -1))
    o_site_mask = (atom_types == type_map.get('O', -1)) | (atom_types == type_map.get('VO', -1))

    return {
        'positions': poscar['LattPnt'],
        'atom_types': atom_types,
        'b_site_mask': b_site_mask,
        'o_site_mask': o_site_mask,
        'element_names': ele_names,
        'type_map': type_map,
        'atom_counts': poscar['AtomNum']
    }


# =============================================================================
# Sublattice Swap Operations
# =============================================================================

def sample_sublattice_swap(
    atom_types: np.ndarray,      # [batch, N]
    sublattice_mask: np.ndarray, # [N] boolean
    type_a: int,
    type_b: int
) -> tuple:
    """
    NumPy sublattice-constrained swap

    Returns:
        swapped_types: [batch, N]
        swap_indices: [batch, 2]
    """
    batch_size, N = atom_types.shape

    # Get sublattice indices
    sub_indices = np.where(sublattice_mask)[0]  # [M]

    # Extract sublattice types
    sub_types = atom_types[:, sub_indices]  # [batch, M]

    # Find type_a and type_b positions
    swapped_types = atom_types.copy()
    swap_indices = np.zeros((batch_size, 2), dtype=np.int32)

    for b in range(batch_size):
        # Get indices of type_a and type_b within sublattice
        type_a_local = np.where(sub_types[b] == type_a)[0]
        type_b_local = np.where(sub_types[b] == type_b)[0]

        # Random selection
        local_a = np.random.choice(type_a_local)
        local_b = np.random.choice(type_b_local)

        # Map to global indices
        global_a = sub_indices[local_a]
        global_b = sub_indices[local_b]

        # Perform swap
        swapped_types[b, global_a] = type_b
        swapped_types[b, global_b] = type_a

        swap_indices[b] = [global_a, global_b]

    return swapped_types, swap_indices


def apply_random_swaps(
    atom_types: np.ndarray,  # [batch, N]
    b_site_mask: np.ndarray,
    o_site_mask: np.ndarray,
    type_map: dict,
    n_swaps: int = 10,
    swap_mode: str = 'both'
) -> tuple:
    """Apply multiple random swaps"""
    current = atom_types.copy()
    swap_history = []

    ti_type = type_map.get('Ti', 1)
    fe_type = type_map.get('Fe', 2)
    o_type = type_map.get('O', 3)
    vo_type = type_map.get('VO', 4)

    for step in range(n_swaps):
        if swap_mode == 'B-site':
            do_b_site = True
        elif swap_mode == 'O-site':
            do_b_site = False
        else:
            do_b_site = np.random.rand() < 0.5

        if do_b_site:
            current, indices = sample_sublattice_swap(
                current, b_site_mask, ti_type, fe_type
            )
        else:
            current, indices = sample_sublattice_swap(
                current, o_site_mask, o_type, vo_type
            )

        swap_history.append(('B' if do_b_site else 'O', indices.copy()))

    return current, swap_history


# =============================================================================
# Visualization
# =============================================================================

def visualize_batch(atom_types: np.ndarray, element_names: list, atom_counts: list):
    """Visualize atom type distribution"""
    batch_size = atom_types.shape[0]

    print("\n" + "="*70)
    print("Atom Type Distribution After Swaps")
    print("="*70)
    print(f"{'Batch':>6} | " + " | ".join(f"{n:>4}" for n in element_names) + " | Status")
    print("-"*70)

    for b in range(batch_size):
        types_b = atom_types[b]
        counts = [(types_b == t).sum() for t in range(len(element_names))]

        status = "✓" if counts == atom_counts else "✗"
        print(f"{b:>6} | " + " | ".join(f"{c:>4}" for c in counts) + f" | {status}")


def visualize_sublattice_detail(
    atom_types: np.ndarray,
    b_site_mask: np.ndarray,
    o_site_mask: np.ndarray,
    type_map: dict,
    batch_idx: int = 0
):
    """Show detailed sublattice configuration"""
    types = atom_types[batch_idx]

    # B-site
    b_indices = np.where(b_site_mask)[0]
    b_types = types[b_indices]
    ti_count = (b_types == type_map['Ti']).sum()
    fe_count = (b_types == type_map['Fe']).sum()

    # O-site
    o_indices = np.where(o_site_mask)[0]
    o_types = types[o_indices]
    o_count = (o_types == type_map['O']).sum()
    vo_count = (o_types == type_map['VO']).sum()

    print(f"\n--- Batch {batch_idx} Sublattice Detail ---")
    print(f"  B-site ({len(b_indices)} sites): Ti={ti_count}, Fe={fe_count}")
    print(f"  O-site ({len(o_indices)} sites): O={o_count}, VO={vo_count}")

    # Show B-site pattern
    print(f"\n  B-site configuration (32 sites):")
    symbols = ['Ti' if t == type_map['Ti'] else 'Fe' for t in b_types]
    for i in range(0, 32, 8):
        row = symbols[i:i+8]
        print(f"    {i:2d}-{i+7:2d}: " + " ".join(f"{s:2s}" for s in row))

    # Show O-site pattern (first 32)
    print(f"\n  O-site configuration (first 32 of 96):")
    o_symbols = ['O ' if t == type_map['O'] else 'VO' for t in o_types[:32]]
    for i in range(0, 32, 8):
        row = o_symbols[i:i+8]
        print(f"    {i:2d}-{i+7:2d}: " + " ".join(f"{s:2s}" for s in row))


def show_swap_history(swap_history: list, batch_idx: int = 0, max_show: int = 10):
    """Show swap history"""
    print(f"\n--- Swap History (first {min(len(swap_history), max_show)} steps) ---")
    for i, (site, indices) in enumerate(swap_history[:max_show]):
        idx_a, idx_b = indices[batch_idx]
        print(f"  Step {i+1:2d}: {site}-site swap at ({idx_a:3d}, {idx_b:3d})")
    if len(swap_history) > max_show:
        print(f"  ... and {len(swap_history) - max_show} more swaps")


# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(42)

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
    print("  Sublattice-Constrained Swap Diffusion Demo")
    print("="*70)

    # 1. Parse POSCAR
    print("\n[1] Parsing POSCAR...")
    poscar = parse_poscar_string(poscar_str)
    print(f"    Elements: {poscar['EleName']}")
    print(f"    Counts:   {poscar['AtomNum']}")
    print(f"    Total:    {poscar['AtomSum']} atoms")

    # 2. Convert to arrays
    print("\n[2] Creating tensor representation...")
    data = poscar_to_arrays(poscar)
    print(f"    Atom types shape: {data['atom_types'].shape}")
    print(f"    B-site atoms: {data['b_site_mask'].sum()} (Ti + Fe)")
    print(f"    O-site atoms: {data['o_site_mask'].sum()} (O + VO)")

    # 3. Create batch of 16
    print("\n[3] Creating batch of 16 structures...")
    batch_size = 16
    N = len(data['atom_types'])
    atom_types_batch = np.tile(data['atom_types'], (batch_size, 1))  # [16, 160]
    print(f"    Batch shape: {atom_types_batch.shape}")

    # Show original configuration
    print("\n[4] Original configuration:")
    visualize_sublattice_detail(
        atom_types_batch,
        data['b_site_mask'],
        data['o_site_mask'],
        data['type_map'],
        batch_idx=0
    )

    # 5. Apply swaps with increasing noise
    print("\n[5] Applying random swaps (2 to 32 per structure)...")
    noisy_batch = atom_types_batch.copy()
    all_histories = []

    for b in range(batch_size):
        n_swaps = (b + 1) * 2  # 2, 4, 6, ..., 32 swaps

        # Apply swaps to single structure
        single = atom_types_batch[b:b+1].copy()
        noisy, history = apply_random_swaps(
            single,
            data['b_site_mask'],
            data['o_site_mask'],
            data['type_map'],
            n_swaps=n_swaps,
            swap_mode='both'
        )
        noisy_batch[b] = noisy[0]
        all_histories.append((n_swaps, history))

    # 6. Visualize results
    visualize_batch(noisy_batch, data['element_names'], data['atom_counts'])

    # 7. Show detailed comparison
    print("\n" + "="*70)
    print("Detailed Comparison: Original vs After 32 swaps")
    print("="*70)

    print("\n>>> Batch 0 (2 swaps - minimal noise):")
    visualize_sublattice_detail(
        noisy_batch, data['b_site_mask'], data['o_site_mask'],
        data['type_map'], batch_idx=0
    )
    show_swap_history(all_histories[0][1], batch_idx=0)

    print("\n>>> Batch 15 (32 swaps - maximum noise):")
    visualize_sublattice_detail(
        noisy_batch, data['b_site_mask'], data['o_site_mask'],
        data['type_map'], batch_idx=15
    )
    show_swap_history(all_histories[15][1], batch_idx=0)

    # 8. Verify composition
    print("\n" + "="*70)
    print("Composition Verification")
    print("="*70)

    original_counts = data['atom_counts']
    all_correct = True

    for b in range(batch_size):
        counts = [(noisy_batch[b] == t).sum() for t in range(len(data['element_names']))]
        if counts != original_counts:
            print(f"  Batch {b}: FAILED")
            all_correct = False

    if all_correct:
        print("  ✓ All 16 structures preserve composition: Sr=32, Ti=16, Fe=16, O=88, VO=8")

    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
