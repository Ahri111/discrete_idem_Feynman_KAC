import numpy as np
from scipy.spatial.transform import Rotation as R
from functools import lru_cache


def generate_full_octahedral_symmetries():
    """
    Generate all 48 octahedral symmetry operations.
    
    Creates permutations of 6 B/O positions corresponding to:
    - 24 rotations (proper rotations)
    - 24 rotoreflections (rotation + inversion)
    
    Returns:
        symmetries: List of 48 permutations, each is a list of 6 indices
                   representing how to permute [B1,B2,B3,B4,B5,B6] positions
    
    Example:
        symmetries[0] might be [0,1,2,3,4,5] (identity)
        symmetries[1] might be [1,0,3,2,5,4] (90° rotation about z)
    """
    # Define 6 directions: [+x, -x, +y, -y, +z, -z]
    vecs = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    
    # Get octahedral rotation group (24 operations)
    group_o = R.create_group('O')
    symmetries = []
    
    # Generate rotations
    for r in group_o:
        perm = []
        for v in vecs:
            rotated = r.apply(v)
            # Find which direction this maps to
            idx = np.argmin(np.sum((vecs - rotated)**2, axis=1))
            perm.append(idx)
        symmetries.append(perm)
    
    # Add inversion operations (24 more)
    # Inversion: [+x,-x,+y,-y,+z,-z] → [-x,+x,-y,+y,-z,+z]
    inversion_map = [1, 0, 3, 2, 5, 4]
    for rot_perm in symmetries[:24]:
        inverted_perm = [inversion_map[rot_perm[i]] for i in range(6)]
        symmetries.append(inverted_perm)
    
    # Remove duplicates (though there shouldn't be any)
    unique_symmetries = []
    for sym in symmetries:
        if sym not in unique_symmetries:
            unique_symmetries.append(sym)
    
    return unique_symmetries


def apply_symmetry(cluster, symmetry_op):
    """
    Apply symmetry operation to a cluster.
    
    Cluster structure:
    - cluster[0]: Core B-site atom
    - cluster[1:7]: 6 neighboring B-sites
    - cluster[7:13]: 6 neighboring O-sites
    - cluster[13:21]: 8 corner A-sites (optional)
    
    Args:
        cluster: List representing cluster configuration
        symmetry_op: Permutation list (length 6) from generate_full_octahedral_symmetries
    
    Returns:
        permuted_cluster: Cluster with permuted neighbor positions
    """
    core = cluster[0]
    
    # Permute B and O positions according to symmetry operation
    b_positions = [cluster[1+i] for i in symmetry_op]
    o_positions = [cluster[7+i] for i in symmetry_op]
    
    # Handle A-site corners if present
    if len(cluster) > 13:
        # A-sites also need permutation (8 cube corners)
        # Map 6-direction symmetry to 8-corner permutation
        a_positions = [cluster[13 + (i % 8)] for i in symmetry_op[:8]]
        return [core] + b_positions + o_positions + a_positions
    
    return [core] + b_positions + o_positions


@lru_cache(maxsize=10000)
def get_canonical_form(cluster_tuple):
    """
    Get canonical (minimum) form of cluster considering all symmetries.
    
    Applies all 48 octahedral symmetries and returns the lexicographically
    smallest representation. This ensures symmetrically equivalent clusters
    map to the same canonical form.
    
    Args:
        cluster_tuple: Tuple representing cluster (must be tuple for caching)
    
    Returns:
        canonical_cluster: Tuple representing canonical form
    
    Example:
        cluster1 = (Ti, Ti, Zr, Zr, Ti, Zr, ...)
        cluster2 = (Ti, Zr, Ti, Zr, Ti, Zr, ...)  # 90° rotation
        → Both return same canonical form
    """
    symmetries = generate_full_octahedral_symmetries()
    
    # Generate all equivalent clusters
    equivalent_clusters = [
        tuple(apply_symmetry(list(cluster_tuple), sym)) 
        for sym in symmetries
    ]
    
    # Return minimum (lexicographic order)
    return min(equivalent_clusters)