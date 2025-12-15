"""
Cluster counting for Cluster Expansion energy evaluation.

Identifies and counts octahedral clusters in perovskite structures
by matching local environments to reference clusters.
"""

import numpy as np
from .structure_utils import posreader, dismatcreate, create_atom_type_mapping
from .reference_generator import load_reference_clusters
from .symmetry import get_canonical_form


def find_positioned_neighbors(core_idx, poscar, atom_types, atom_ind_group):
    """
    Find and position neighbors around a B-site core atom.
    
    Searches for atoms within distance ranges and assigns them to
    specific directional positions (6 B/O positions + 8 A corners).
    
    Args:
        core_idx: Index of core B-site atom
        poscar: Structure dictionary with 'dismat' field
        atom_types: List of atom type indices for each atom
        atom_ind_group: [[A_types], [B_types], [O_types]]
    
    Returns:
        positioned_neighbors: Dictionary with keys:
            - 'b_positions': List of 6 B-site atom types (or None)
            - 'o_positions': List of 6 O-site atom types (or None)
            - 'a_positions': List of 8 A-site atom types (or None)
    """
    positions = [np.array(pos) for pos in poscar['LattPnt']]
    distances = poscar['dismat'][core_idx]
    core_pos = positions[core_idx]
    
    # Distance ranges (Angstroms) - tuned for perovskites
    b_range = (3.8, 4.2)  # B-B nearest neighbor distance
    o_range = (1.8, 2.2)  # B-O distance
    a_range = (3.0, 4.0)  # B-A distance
    
    # Initialize empty positions
    positioned_neighbors = {
        'b_positions': [None] * 6,
        'o_positions': [None] * 6,
        'a_positions': [None] * 8
    }
    
    # Find candidates
    candidates = {'b': [], 'o': [], 'a': []}
    
    for i, dist in enumerate(distances):
        if i == core_idx:
            continue
        
        atom_type = atom_types[i]
        
        # Compute direction vector (with PBC)
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
        assign_b_o_direction(cand, positioned_neighbors['b_positions'])
    
    for cand in candidates['o'][:6]:
        assign_b_o_direction(cand, positioned_neighbors['o_positions'])
    
    for cand in candidates['a'][:8]:
        assign_a_direction(cand, positioned_neighbors['a_positions'])
    
    return positioned_neighbors


def assign_b_o_direction(candidate, position_array):
    """
    Assign B or O neighbor to one of 6 octahedral directions.
    
    Directions:
    - 0: +x+y quadrant (xy plane)
    - 1: -x-y quadrant
    - 2: +x-y quadrant
    - 3: -x+y quadrant
    - 4: +z direction
    - 5: -z direction
    
    Args:
        candidate: Tuple of (atom_type, direction_vector)
        position_array: List of 6 positions to fill (modified in-place)
    """
    atom_type, vec = candidate
    x, y, z = vec
    abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)
    
    # Check if z-direction is dominant
    if abs_z > max(abs_x, abs_y) * 1.2:
        idx = 4 if z > 0 else 5
    else:
        # Assign to xy-plane quadrant
        if x >= 0 and y >= 0:
            idx = 0  # Quadrant 1
        elif x < 0 and y < 0:
            idx = 1  # Quadrant 3
        elif x >= 0 and y < 0:
            idx = 2  # Quadrant 4
        else:
            idx = 3  # Quadrant 2
    
    if position_array[idx] is None:
        position_array[idx] = atom_type


def assign_a_direction(candidate, position_array):
    """
    Assign A-site neighbor to one of 8 cube corners.
    
    Cube corners indexed by binary: (x>0)(y>0)(z>0)
    - 0: (-x, -y, -z)
    - 1: (+x, -y, -z)
    - 2: (-x, +y, -z)
    - 3: (+x, +y, -z)
    - 4: (-x, -y, +z)
    - 5: (+x, -y, +z)
    - 6: (-x, +y, +z)
    - 7: (+x, +y, +z)
    
    Args:
        candidate: Tuple of (atom_type, direction_vector)
        position_array: List of 8 positions to fill (modified in-place)
    """
    atom_type, vec = candidate
    x, y, z = vec
    
    idx = (1 if x > 0 else 0) + (2 if y > 0 else 0) + (4 if z > 0 else 0)
    
    if position_array[idx] is None:
        position_array[idx] = atom_type


def generate_single_positioned_cluster(core_type, positioned_neighbors):
    """
    Generate cluster from positioned neighbors.
    
    Fills missing positions with most common neighbor type and
    validates that cluster has sufficient neighbors.
    
    Args:
        core_type: Atom type of core B-site
        positioned_neighbors: Dict from find_positioned_neighbors
    
    Returns:
        cluster: List representing cluster, or None if invalid
    """
    b_positions = positioned_neighbors['b_positions']
    o_positions = positioned_neighbors['o_positions']
    a_positions = positioned_neighbors.get('a_positions', [])
    
    # Check validity (need at least 4 B and 4 O neighbors)
    valid_b = sum(1 for x in b_positions if x is not None)
    valid_o = sum(1 for x in o_positions if x is not None)
    
    if valid_b < 4 or valid_o < 4:
        return None
    
    # Fill missing positions with most common type
    def fill_none_values(positions):
        filled = positions.copy()
        valid_types = [x for x in positions if x is not None]
        if valid_types:
            most_common = max(set(valid_types), key=valid_types.count)
            filled = [most_common if x is None else x for x in filled]
        return filled
    
    filled_b = fill_none_values(b_positions)
    filled_o = fill_none_values(o_positions)
    
    # Include A-sites if present
    if a_positions and any(x is not None for x in a_positions):
        filled_a = fill_none_values(a_positions)
        return [core_type] + filled_b + filled_o + filled_a
    
    return [core_type] + filled_b + filled_o


def count_clusters_in_structure(file_poscar, atom_ind_group, reference_file):
    """
    Count all clusters in a structure.
    
    Args:
        file_poscar: Path to POSCAR file
        atom_ind_group: [[A_types], [B_types], [O_types]]
        reference_file: Path to reference clusters JSON
    
    Returns:
        cluster_counts: List of counts for each reference cluster
        reference_clusters: List of reference cluster tuples
    """
    reference_clusters = load_reference_clusters(reference_file)
    
    # Load structure
    poscar = posreader(file_poscar)
    poscar = dismatcreate(poscar)
    atom_types = create_atom_type_mapping(poscar)
    
    # Find all B-site atoms
    b_site_indices = [i for i, t in enumerate(atom_types) if t in atom_ind_group[1]]
    
    # Initialize counts
    cluster_counts = [0] * len(reference_clusters)
    
    # Count clusters at each B-site
    for b_idx in b_site_indices:
        core_type = atom_types[b_idx]
        
        # Find neighbors
        positioned_neighbors = find_positioned_neighbors(
            b_idx, poscar, atom_types, atom_ind_group
        )
        
        # Generate cluster
        cluster = generate_single_positioned_cluster(core_type, positioned_neighbors)
        
        if cluster:
            # Get canonical form
            canonical = get_canonical_form(tuple(cluster))
            
            # Match to reference
            try:
                ref_idx = reference_clusters.index(canonical)
                cluster_counts[ref_idx] += 1
            except ValueError:
                # Cluster not in reference (skip)
                pass
    
    return cluster_counts, reference_clusters


def count_cluster(file_poscar, atom_ind_group, cluster_dir, verbose=False):
    """
    Main interface for cluster counting.
    
    Args:
        file_poscar: Path to POSCAR file
        atom_ind_group: [[A_types], [B_types], [O_types]]
        cluster_dir: Path to reference clusters JSON
        verbose: If True, print detailed statistics
    
    Returns:
        result_counts: [Ti_count, O_count, cluster1, cluster2, ...]
        reference_clusters: List of reference cluster tuples
    """
    cluster_counts, reference_clusters = count_clusters_in_structure(
        file_poscar, atom_ind_group, reference_file=cluster_dir
    )
    
    if cluster_counts:
        poscar = posreader(file_poscar)
        atom_types = create_atom_type_mapping(poscar)
        
        # Count Ti and O atoms (hardcoded for now - TODO: generalize)
        ti_count = sum(1 for atom_type in atom_types if atom_type == 1)
        o_count = sum(1 for atom_type in atom_types if atom_type == 3)
        
        result_counts = [ti_count, o_count] + cluster_counts
        
        if verbose:
            print(f"\nRESULTS (Oh Symmetry):")
            print(f"Ti atom count: {ti_count}")
            print(f"O atom count: {o_count}")
            print(f"Reference clusters: {len(reference_clusters)}")
            print(f"Total clusters found: {sum(cluster_counts)}")
            print(f"Non-zero cluster types: {sum(1 for c in cluster_counts if c > 0)}")
    else:
        result_counts = None
    
    return result_counts, reference_clusters