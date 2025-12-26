"""
In-memory cluster counting and energy evaluation (no file I/O).

Optimized for MCMC sampling by avoiding disk operations.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from src.energy_models.cluster_expansion.structure_utils import create_atom_type_mapping
from src.energy_models.cluster_expansion.symmetry import get_canonical_form


def build_atom_ind_group(atom_group: List[List[str]], poscar: Dict) -> List[List[int]]:
    """
    Convert atom element names to atom type indices.

    Original logic from reference code:
    for sub, group in enumerate(atom_group):
        atom_ind_group.append([])
        for i in range(len(group)):
            atom_ind_group[sub].append(index)
            index += 1

    Args:
        atom_group: List of element name groups, e.g., [['Sr', 'La'], ['Ti'], ['O']]
        poscar: Structure dictionary with 'EleName' field

    Returns:
        atom_ind_group: List of type index groups, e.g., [[0, 1], [2], [3]]

    Example:
        atom_group = [['Sr', 'La'], ['Ti'], ['O']]
        poscar['EleName'] = ['Sr', 'La', 'Ti', 'O']
        → atom_ind_group = [[0, 1], [2], [3]]
    """
    ele_to_idx = {name: idx for idx, name in enumerate(poscar['EleName'])}

    atom_ind_group = []
    for group in atom_group:
        indices = []
        for elem in group:
            if elem in ele_to_idx:
                indices.append(ele_to_idx[elem])
            else:
                raise ValueError(f"Element '{elem}' not found in POSCAR")
        atom_ind_group.append(indices)

    return atom_ind_group


def compute_distance_matrix_inplace(positions: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """
    Compute distance matrix with periodic boundary conditions (optimized).

    Args:
        positions: (N, 3) array of fractional coordinates
        lattice: (3, 3) lattice vectors

    Returns:
        dismat: (N, N) distance matrix
    """
    N = len(positions)
    dismat = np.zeros((N, N))

    # Vectorized computation
    for i in range(N):
        delta = positions - positions[i]

        # Minimum image convention
        delta = np.where(delta > 0.5, delta - 1, delta)
        delta = np.where(delta <= -0.5, delta + 1, delta)

        # Convert to Cartesian
        cart_delta = np.dot(np.abs(delta), lattice)
        dismat[i] = np.linalg.norm(cart_delta, axis=1)

    return dismat


def find_neighbors_inplace(
    core_idx: int,
    positions: np.ndarray,
    dismat: np.ndarray,
    atom_types: List[int],
    atom_ind_group: List[List[int]],
    lattice: np.ndarray
) -> Dict:
    """
    Find positioned neighbors without file I/O.

    Args:
        core_idx: Index of core B-site atom
        positions: (N, 3) fractional coordinates
        dismat: (N, N) distance matrix
        atom_types: List of atom type indices
        atom_ind_group: [[A_types], [B_types], [O_types]]
        lattice: (3, 3) lattice vectors

    Returns:
        positioned_neighbors: Dict with 'b_positions', 'o_positions', 'a_positions'
    """
    core_pos = positions[core_idx]
    distances = dismat[core_idx]

    # Distance ranges (Angstroms)
    b_range = (3.8, 4.2)
    o_range = (1.8, 2.2)
    a_range = (3.0, 4.0)

    positioned_neighbors = {
        'b_positions': [None] * 6,
        'o_positions': [None] * 6,
        'a_positions': [None] * 8
    }

    candidates = {'b': [], 'o': [], 'a': []}

    for i, dist in enumerate(distances):
        if i == core_idx:
            continue

        atom_type = atom_types[i]

        # Compute direction vector
        vec = positions[i] - core_pos
        vec = np.where(vec > 0.5, vec - 1, vec)
        vec = np.where(vec <= -0.5, vec + 1, vec)

        # Classify by type and distance
        if atom_type in atom_ind_group[1] and b_range[0] <= dist <= b_range[1]:
            candidates['b'].append((atom_type, vec))
        elif atom_type in atom_ind_group[2] and o_range[0] <= dist <= o_range[1]:
            candidates['o'].append((atom_type, vec))
        elif len(atom_ind_group) > 0 and atom_type in atom_ind_group[0] and a_range[0] <= dist <= a_range[1]:
            candidates['a'].append((atom_type, vec))

    # Assign to directional positions
    for cand in candidates['b'][:6]:
        _assign_b_o_direction(cand, positioned_neighbors['b_positions'])

    for cand in candidates['o'][:6]:
        _assign_b_o_direction(cand, positioned_neighbors['o_positions'])

    for cand in candidates['a'][:8]:
        _assign_a_direction(cand, positioned_neighbors['a_positions'])

    return positioned_neighbors


def _assign_b_o_direction(candidate: Tuple, position_array: List):
    """Assign B or O neighbor to octahedral direction."""
    atom_type, vec = candidate
    x, y, z = vec
    abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)

    if abs_z > max(abs_x, abs_y) * 1.2:
        idx = 4 if z > 0 else 5
    else:
        if x >= 0 and y >= 0:
            idx = 0
        elif x < 0 and y < 0:
            idx = 1
        elif x >= 0 and y < 0:
            idx = 2
        else:
            idx = 3

    if position_array[idx] is None:
        position_array[idx] = atom_type


def _assign_a_direction(candidate: Tuple, position_array: List):
    """Assign A-site neighbor to cube corner."""
    atom_type, vec = candidate
    x, y, z = vec

    idx = (1 if x > 0 else 0) + (2 if y > 0 else 0) + (4 if z > 0 else 0)

    if position_array[idx] is None:
        position_array[idx] = atom_type


def generate_cluster_from_neighbors(core_type: int, positioned_neighbors: Dict) -> Optional[List[int]]:
    """Generate cluster from positioned neighbors."""
    b_positions = positioned_neighbors['b_positions']
    o_positions = positioned_neighbors['o_positions']
    a_positions = positioned_neighbors.get('a_positions', [])

    # Validate
    valid_b = sum(1 for x in b_positions if x is not None)
    valid_o = sum(1 for x in o_positions if x is not None)

    if valid_b < 4 or valid_o < 4:
        return None

    # Fill missing positions
    def fill_none_values(positions):
        filled = positions.copy()
        valid_types = [x for x in positions if x is not None]
        if valid_types:
            most_common = max(set(valid_types), key=valid_types.count)
            filled = [most_common if x is None else x for x in filled]
        return filled

    filled_b = fill_none_values(b_positions)
    filled_o = fill_none_values(o_positions)

    if a_positions and any(x is not None for x in a_positions):
        filled_a = fill_none_values(a_positions)
        return [core_type] + filled_b + filled_o + filled_a

    return [core_type] + filled_b + filled_o


def count_clusters_from_structure_inplace(
    positions: np.ndarray,
    lattice: np.ndarray,
    atom_types: List[int],
    atom_ind_group: List[List[int]],
    reference_clusters: List[Tuple]
) -> np.ndarray:
    """
    Count clusters directly from structure arrays (no file I/O).

    This is the optimized version for MCMC sampling.

    Args:
        positions: (N, 3) fractional coordinates
        lattice: (3, 3) lattice vectors
        atom_types: List of atom type indices
        atom_ind_group: [[A_types], [B_types], [O_types]]
        reference_clusters: List of reference cluster tuples

    Returns:
        cluster_counts: Array of cluster counts
    """
    # Compute distance matrix once
    dismat = compute_distance_matrix_inplace(positions, lattice)

    # Find B-site atoms
    b_site_indices = [i for i, t in enumerate(atom_types) if t in atom_ind_group[1]]

    # Initialize counts
    cluster_counts = np.zeros(len(reference_clusters), dtype=np.int32)

    # Count clusters at each B-site
    for b_idx in b_site_indices:
        core_type = atom_types[b_idx]

        # Find neighbors
        positioned_neighbors = find_neighbors_inplace(
            b_idx, positions, dismat, atom_types, atom_ind_group, lattice
        )

        # Generate cluster
        cluster = generate_cluster_from_neighbors(core_type, positioned_neighbors)

        if cluster:
            # Get canonical form
            canonical = get_canonical_form(tuple(cluster))

            # Match to reference
            try:
                ref_idx = reference_clusters.index(canonical)
                cluster_counts[ref_idx] += 1
            except ValueError:
                pass

    return cluster_counts


def compute_features_from_structure_inplace(
    positions: np.ndarray,
    lattice: np.ndarray,
    atom_types: List[int],
    atom_ind_group: List[List[int]],
    reference_clusters: List[Tuple]
) -> np.ndarray:
    """
    Compute full feature vector from structure (no file I/O).

    Returns feature vector: [Ti_count, O_count, cluster1, cluster2, ...]

    Args:
        positions: (N, 3) fractional coordinates
        lattice: (3, 3) lattice vectors
        atom_types: List of atom type indices
        atom_ind_group: [[A_types], [B_types], [O_types]]
        reference_clusters: List of reference cluster tuples

    Returns:
        features: Feature vector for energy prediction
    """
    cluster_counts = count_clusters_from_structure_inplace(
        positions, lattice, atom_types, atom_ind_group, reference_clusters
    )

    # Count specific atom types (Ti and O for perovskites)
    # This is hardcoded for now - could be made more general
    ti_count = sum(1 for t in atom_types if t in atom_ind_group[1])  # B-site
    o_count = sum(1 for t in atom_types if t in atom_ind_group[2])   # O-site

    features = np.concatenate([[ti_count, o_count], cluster_counts])

    return features
