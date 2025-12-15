
"""
Reference cluster generation for Cluster Expansion.

Generates all possible unique octahedral clusters considering
compositional constraints and symmetry equivalence.
"""

import json
from itertools import product
from .symmetry import get_canonical_form


def generate_reference_clusters(atom_ind_group):
    """
    Generate all unique reference clusters for a given composition.
    
    Creates all possible combinations of atom types at each position,
    then reduces to unique clusters using octahedral symmetry.
    
    Args:
        atom_ind_group: List of atom type groups
            [0]: A-site atom types (e.g., [0, 2] for Sr, La)
            [1]: B-site atom types (e.g., [1] for Ti)
            [2]: O-site atom types (e.g., [3] for O)
    
    Returns:
        cluster_list: List of unique cluster tuples
        
    Example:
        For SrTiO3 with Sr/La substitution:
        atom_ind_group = [[0, 2], [1], [3]]
        
        Generates clusters like:
        - (Ti, Ti,Ti,Ti,Ti,Ti,Ti, O,O,O,O,O,O, Sr,Sr,Sr,Sr,Sr,Sr,Sr,Sr)
        - (Ti, Ti,Ti,Ti,Ti,Ti,Ti, O,O,O,O,O,O, La,Sr,Sr,Sr,Sr,Sr,Sr,Sr)
        - ...
        
        256 combinations → ~30 unique after symmetry reduction
    """
    b_types = atom_ind_group[1]
    o_types = atom_ind_group[2]
    a_types = atom_ind_group[0] if len(atom_ind_group) > 0 else []
    
    cluster_set = set()
    
    if a_types:
        # Full ABO3 with A-site substitution
        for core_type in b_types:
            for b_combo in product(b_types, repeat=6):
                for o_combo in product(o_types, repeat=6):
                    for a_combo in product(a_types, repeat=8):
                        # Create cluster: [core, 6 B's, 6 O's, 8 A's]
                        cluster = [core_type] + list(b_combo) + list(o_combo) + list(a_combo)
                        
                        # Get canonical form (symmetry reduction)
                        canonical = get_canonical_form(tuple(cluster))
                        cluster_set.add(canonical)
    else:
        # Simple BO3 without A-site
        for core_type in b_types:
            for b_combo in product(b_types, repeat=6):
                for o_combo in product(o_types, repeat=6):
                    cluster = [core_type] + list(b_combo) + list(o_combo)
                    canonical = get_canonical_form(tuple(cluster))
                    cluster_set.add(canonical)
    
    return list(cluster_set)


def save_reference_clusters(clusters, file_path):
    """
    Save reference clusters to JSON file.
    
    Args:
        clusters: List of cluster tuples
        file_path: Output JSON file path
    """
    # Convert tuples to lists for JSON serialization
    cluster_lists = [list(c) for c in clusters]
    
    with open(file_path, 'w') as f:
        json.dump(cluster_lists, f, indent=2)
    
    print(f"Saved {len(clusters)} reference clusters to {file_path}")


def load_reference_clusters(file_path):
    """
    Load reference clusters from JSON file.
    
    Args:
        file_path: JSON file path
    
    Returns:
        clusters: List of cluster tuples
    """
    with open(file_path, 'r') as f:
        cluster_lists = json.load(f)
    
    # Convert lists back to tuples
    clusters = [tuple(c) for c in cluster_lists]
    
    print(f"Loaded {len(clusters)} reference clusters from {file_path}")
    return clusters


if __name__ == "__main__":
    """
    Example usage: Generate and save reference clusters
    """
    # Example: SrTiO3 with Sr/La substitution
    atom_ind_group = [
        [0, 2],  # A-site: Sr(0), La(2)
        [1],     # B-site: Ti(1)
        [3]      # O-site: O(3)
    ]
    
    print("Generating reference clusters...")
    clusters = generate_reference_clusters(atom_ind_group)
    print(f"Generated {len(clusters)} unique clusters")
    
    # Save to file
    save_reference_clusters(clusters, "reference_clusters.json")
    
    # Example cluster
    print("\nExample cluster:")
    print(f"  {clusters[0]}")
    print(f"  Structure: [B_core, 6 B-neighbors, 6 O-neighbors, 8 A-corners]")