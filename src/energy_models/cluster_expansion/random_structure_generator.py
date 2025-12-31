"""
Random Structure Generator for Diffusion Model xT

Generates random atomic configurations with:
- Fixed composition (stoichiometry)
- Fixed lattice sites (A/B/O positions)
- Physical constraints preserved
- Random shuffling within each site type

Usage:
    composition = {
        'A': {'Sr': 24, 'La': 8},
        'B': {'Ti': 24, 'Fe': 8},
        'O': {'O': 88, 'VO': 4}
    }

    structure = generate_random_structure(
        template_file='POSCAR_ABO3',
        composition=composition
    )
"""

import numpy as np
import random
import copy
from src.energy_models.cluster_expansion.structure_utils import posreader, poswriter, dismatcreate


def generate_random_structure(template_file, composition, element_names=None):
    """
    Generate random structure with fixed composition.

    This creates a high-temperature disordered structure (xT for diffusion)
    by randomly assigning atoms within each site type.

    Args:
        template_file: Path to template POSCAR (defines lattice sites)
        composition: Dict of site compositions
                    {'A': {'Sr': 24, 'La': 8},
                     'B': {'Ti': 24, 'Fe': 8},
                     'O': {'O': 88, 'VO': 4}}
        element_names: Optional list of element names for POSCAR
                      If None, auto-generate from composition

    Returns:
        poscar: Structure dict with randomized atom positions
    """
    # Read template structure
    template = posreader(template_file)

    # Validate composition
    _validate_composition(template, composition)

    # Generate atom type assignments
    atom_types = _generate_atom_types(composition)

    # Shuffle within each site type
    atom_types_shuffled = _shuffle_within_sites(atom_types, composition)

    # Create new POSCAR structure
    poscar = _create_poscar_from_types(template, atom_types_shuffled, element_names)

    # Create distance matrix
    poscar = dismatcreate(poscar)

    return poscar


def generate_random_structures_batch(template_file, composition, n_samples,
                                     element_names=None, seed=None):
    """
    Generate multiple random structures.

    Args:
        template_file: Path to template POSCAR
        composition: Composition dict
        n_samples: Number of structures to generate
        element_names: Optional element names
        seed: Random seed for reproducibility

    Returns:
        structures: List of poscar dicts
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    structures = []
    for i in range(n_samples):
        structure = generate_random_structure(template_file, composition, element_names)
        structures.append(structure)

    return structures


def composition_from_ratios(site_ratios, total_atoms_per_site):
    """
    Create composition dict from ratios.

    Args:
        site_ratios: Dict of element ratios per site
                    {'A': {'Sr': 3, 'La': 1},
                     'B': {'Ti': 3, 'Fe': 1},
                     'O': {'O': 23, 'VO': 1}}
        total_atoms_per_site: Dict of total atoms per site
                             {'A': 32, 'B': 32, 'O': 96}

    Returns:
        composition: Absolute composition dict
    """
    composition = {}

    for site, ratios in site_ratios.items():
        total = total_atoms_per_site[site]
        ratio_sum = sum(ratios.values())

        composition[site] = {}
        allocated = 0

        # Allocate atoms proportionally
        items = list(ratios.items())
        for i, (elem, ratio) in enumerate(items[:-1]):
            count = int(round(total * ratio / ratio_sum))
            composition[site][elem] = count
            allocated += count

        # Last element gets remainder
        last_elem = items[-1][0]
        composition[site][last_elem] = total - allocated

    return composition


def _validate_composition(template, composition):
    """
    Validate that composition matches template structure.

    Args:
        template: Template POSCAR dict
        composition: Composition dict
    """
    # Count atoms in each site from composition
    a_count = sum(composition.get('A', {}).values())
    b_count = sum(composition.get('B', {}).values())
    o_count = sum(composition.get('O', {}).values())

    total_comp = a_count + b_count + o_count
    total_template = template['AtomSum']

    if total_comp != total_template:
        raise ValueError(
            f"Composition total ({total_comp}) doesn't match template ({total_template})"
        )


def _generate_atom_types(composition):
    """
    Generate atom type array from composition.

    Args:
        composition: Composition dict

    Returns:
        atom_types: Dict with site keys and type arrays
                   {'A': [0,0,...,2,2], 'B': [1,1,...,3,3], 'O': [4,4,...,5,5]}
    """
    atom_types = {}

    # Element name to type index mapping
    elem_to_type = {}
    type_idx = 0

    for site in ['A', 'B', 'O']:
        if site not in composition:
            continue

        site_types = []
        for elem, count in composition[site].items():
            if elem not in elem_to_type:
                elem_to_type[elem] = type_idx
                type_idx += 1

            site_types.extend([elem_to_type[elem]] * count)

        atom_types[site] = site_types

    return atom_types


def _shuffle_within_sites(atom_types, composition):
    """
    Shuffle atom types within each site independently.

    Args:
        atom_types: Dict of site type arrays
        composition: Composition dict

    Returns:
        shuffled: Shuffled atom types dict
    """
    shuffled = {}

    for site, types in atom_types.items():
        types_copy = types.copy()
        random.shuffle(types_copy)
        shuffled[site] = types_copy

    return shuffled


def _create_poscar_from_types(template, atom_types_shuffled, element_names):
    """
    Create POSCAR structure from shuffled atom types.

    Args:
        template: Template POSCAR dict
        atom_types_shuffled: Shuffled atom types
        element_names: Optional element names

    Returns:
        poscar: New POSCAR dict
    """
    poscar = copy.deepcopy(template)

    # Concatenate all atom types in order: A, B, O
    all_types = []
    for site in ['A', 'B', 'O']:
        if site in atom_types_shuffled:
            all_types.extend(atom_types_shuffled[site])

    # Count atoms of each type
    unique_types = sorted(set(all_types))
    atom_counts = [all_types.count(t) for t in unique_types]

    # Generate element names if not provided
    if element_names is None:
        element_names = [f'Elem{t}' for t in unique_types]

    # Update POSCAR
    poscar['EleName'] = element_names
    poscar['EleNum'] = len(element_names)
    poscar['AtomNum'] = atom_counts

    # Coordinates stay the same (lattice positions fixed)
    # Only atom types change

    return poscar


def save_random_structures(structures, output_dir, prefix='random_struct'):
    """
    Save multiple random structures to files.

    Args:
        structures: List of POSCAR dicts
        output_dir: Output directory
        prefix: Filename prefix
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for i, structure in enumerate(structures):
        filename = os.path.join(output_dir, f'{prefix}_{i:05d}.vasp')
        poswriter(filename, structure)

    print(f"Saved {len(structures)} structures to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Define composition
    composition = {
        'A': {'Sr': 24, 'La': 8},      # 32 A-sites total
        'B': {'Ti': 24, 'Fe': 8},      # 32 B-sites total
        'O': {'O': 88, 'VO': 4}        # 92 O-sites total
    }

    element_names = ['Sr', 'Ti', 'La', 'Fe', 'O', 'VO']

    # Generate single structure
    structure = generate_random_structure(
        template_file='src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3',
        composition=composition,
        element_names=element_names
    )

    print(f"Generated random structure:")
    print(f"  Element names: {structure['EleName']}")
    print(f"  Atom counts: {structure['AtomNum']}")
    print(f"  Total atoms: {structure['AtomSum']}")

    # Generate batch
    structures = generate_random_structures_batch(
        template_file='src/energy_models/cluster_expansion/energy_parameter/POSCAR_ABO3',
        composition=composition,
        n_samples=10,
        element_names=element_names,
        seed=42
    )

    print(f"\nGenerated {len(structures)} random structures")

    # Save to files
    # save_random_structures(structures, 'random_structures/', prefix='xT')
