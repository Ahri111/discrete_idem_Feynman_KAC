"""
Optimized Cluster Expansion Energy Calculator

Tier 1 + 2 Optimizations:
- In-memory processing (no file I/O)
- Reference cluster dict for O(1) lookup
- Incremental cluster update for swaps
- Batch parallel processing
"""

import pickle
import numpy as np
from multiprocessing import Pool
from functools import lru_cache
from src.energy_models.cluster_expansion.structure_utils import (
    posreader, dismatcreate, dismatswap, create_atom_type_mapping
)
from src.energy_models.cluster_expansion.cluster_counter import (
    find_positioned_neighbors, generate_single_positioned_cluster
)
from src.energy_models.cluster_expansion.reference_generator import load_reference_clusters
from src.energy_models.cluster_expansion.symmetry import get_canonical_form
from src.energy_models.cluster_expansion.structure_converter import StructureConverter


class EnergyCalculator:
    """
    Optimized Cluster-Expansion Energy Calculator

    Features:
    - In-memory processing (no temp files)
    - O(1) reference cluster lookup
    - Incremental updates for atom swaps
    - Parallel batch processing

    Args:
        model_file: Path to trained LASSO model (.pkl)
        scaler_file: Path to trained scaler (.pkl)
        cluster_file: Path to reference clusters (.json)
        atom_ind_group: [[A_types], [B_types], [O_types]]
                       e.g., [[0, 2], [1, 3], [4, 5]]
    """

    def __init__(self, model_file, scaler_file, cluster_file, atom_ind_group,
                 element_names=None, template_file=None):
        # Load LASSO model
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

        # Load scaler
        with open(scaler_file, "rb") as f:
            self.scaler = pickle.load(f)

        # Load reference clusters as list
        self.reference_clusters = load_reference_clusters(cluster_file)

        # Create dict for O(1) lookup (Tier 1 optimization)
        self.cluster_to_idx = {
            cluster: idx for idx, cluster in enumerate(self.reference_clusters)
        }

        self.atom_ind_group = atom_ind_group

        # Cache for incremental updates
        self._cluster_cache = {}  # {structure_id: {b_idx: cluster}}

        # Structure converter for tensor support
        self.converter = StructureConverter(
            element_names=element_names,
            template_file=template_file
        )

        print(f"[EnergyCalculator] Loaded model with {len(self.reference_clusters)} reference clusters")
        print(f"[EnergyCalculator] LASSO features: {np.count_nonzero(self.model.coef_)}")


    def compute_energy(self, poscar, use_cache=False, structure_id=None):
        """
        Compute energy for a single structure.

        Args:
            poscar: Structure dict from posreader() with 'dismat' field
            use_cache: Whether to use cached clusters (for incremental updates)
            structure_id: Optional ID for caching

        Returns:
            energy: Predicted energy (float)
        """
        # Count clusters
        cluster_list = self._count_clusters_from_poscar(
            poscar, use_cache=use_cache, structure_id=structure_id
        )

        # Scale and predict
        cluster_scaled = self.scaler.transform([cluster_list])[0]
        energy = self.model.predict([cluster_scaled])[0]

        return energy


    def compute_energy_batch(self, poscar_list, n_workers=8):
        """
        Compute energies for multiple structures in parallel.

        Args:
            poscar_list: List of structure dicts
            n_workers: Number of parallel workers

        Returns:
            energies: Numpy array of shape [len(poscar_list)]
        """
        # Parallel cluster counting
        with Pool(n_workers) as pool:
            cluster_lists = pool.starmap(
                self._count_clusters_from_poscar,
                [(poscar, False, None) for poscar in poscar_list]
            )

        # Batch predict (LASSO is fast)
        cluster_scaled = self.scaler.transform(cluster_lists)
        energies = self.model.predict(cluster_scaled)

        return energies


    def compute_energy_incremental(self, poscar, swapped_idx1, swapped_idx2,
                                   structure_id='default'):
        """
        Compute energy after swapping two atoms (incremental update).

        This is MUCH faster than full recomputation because only
        affected clusters are recomputed.

        Args:
            poscar: Structure dict AFTER swap (with updated dismat)
            swapped_idx1, swapped_idx2: Indices of swapped atoms
            structure_id: ID for cache lookup

        Returns:
            energy: Predicted energy
        """
        # Find affected B-sites
        affected_sites = self._find_affected_sites(
            poscar, swapped_idx1, swapped_idx2
        )

        # Update only affected clusters
        cluster_list = self._incremental_update(
            poscar, affected_sites, structure_id
        )

        # Scale and predict
        cluster_scaled = self.scaler.transform([cluster_list])[0]
        energy = self.model.predict([cluster_scaled])[0]

        return energy


    def compute_energy_from_tensor(self, positions, atom_types, lattice, metadata=None):
        """
        Compute energy from PyTorch tensors (for diffusion model).

        Args:
            positions: [N, 3] tensor of fractional coordinates
            atom_types: [N] tensor of type indices
            lattice: [3, 3] tensor of lattice vectors
            metadata: Optional metadata dict

        Returns:
            energy: Predicted energy (float)
        """
        # Convert to POSCAR
        poscar = self.converter.tensor_to_poscar(positions, atom_types, lattice, metadata)

        # Add distance matrix
        poscar = dismatcreate(poscar)

        # Compute energy
        energy = self.compute_energy(poscar)

        return energy


    def compute_energy_batch_from_tensor(self, positions, atom_types, lattice,
                                         metadata=None, n_workers=8):
        """
        Compute energies from batched tensors (for diffusion model).

        Args:
            positions: [B, N, 3] tensor
            atom_types: [B, N] tensor
            lattice: [3, 3] or [B, 3, 3] tensor
            metadata: Optional list of metadata dicts
            n_workers: Number of parallel workers

        Returns:
            energies: [B] numpy array of energies
        """
        # Convert to POSCAR list
        poscar_list = self.converter.batch_tensor_to_poscar(
            positions, atom_types, lattice, metadata
        )

        # Add distance matrices
        poscar_list = [dismatcreate(p) for p in poscar_list]

        # Batch compute
        energies = self.compute_energy_batch(poscar_list, n_workers=n_workers)

        return energies


    def _count_clusters_from_poscar(self, poscar, use_cache=False, structure_id=None):
        """
        Count clusters from POSCAR dict (in-memory, no file I/O).

        Args:
            poscar: Structure dict with 'dismat' field
            use_cache: Use cached clusters if available
            structure_id: Cache identifier

        Returns:
            cluster_list: [Ti_count, O_count, cluster1_count, ..., clusterN_count]
        """
        atom_types = create_atom_type_mapping(poscar)
        b_site_indices = [i for i, t in enumerate(atom_types) if t in self.atom_ind_group[1]]

        # Initialize cluster counts
        cluster_counts = [0] * len(self.reference_clusters)

        # Initialize cache if needed
        if use_cache and structure_id:
            if structure_id not in self._cluster_cache:
                self._cluster_cache[structure_id] = {}
            cluster_cache = self._cluster_cache[structure_id]
        else:
            cluster_cache = None

        # Count clusters at each B-site
        for b_idx in b_site_indices:
            # Check cache
            if cluster_cache is not None and b_idx in cluster_cache:
                canonical = cluster_cache[b_idx]
            else:
                # Compute cluster
                core_type = atom_types[b_idx]
                positioned_neighbors = find_positioned_neighbors(
                    b_idx, poscar, atom_types, self.atom_ind_group
                )
                cluster = generate_single_positioned_cluster(core_type, positioned_neighbors)

                if cluster is None:
                    continue

                canonical = get_canonical_form(tuple(cluster))

                # Cache it
                if cluster_cache is not None:
                    cluster_cache[b_idx] = canonical

            # Lookup in reference (O(1) with dict)
            ref_idx = self.cluster_to_idx.get(canonical, None)
            if ref_idx is not None:
                cluster_counts[ref_idx] += 1

        # Prepend B-site and O-site total counts
        b_count = sum(1 for t in atom_types if t in self.atom_ind_group[1])
        o_count = sum(1 for t in atom_types if t in self.atom_ind_group[2])

        result = [b_count, o_count] + cluster_counts

        return result


    def _find_affected_sites(self, poscar, swapped_idx1, swapped_idx2):
        """
        Find B-sites affected by atom swap.

        A B-site is affected if either swapped atom is within neighbor range.

        Args:
            poscar: Structure dict with 'dismat'
            swapped_idx1, swapped_idx2: Swapped atom indices

        Returns:
            affected_sites: Set of B-site indices
        """
        atom_types = create_atom_type_mapping(poscar)
        b_site_indices = [i for i, t in enumerate(atom_types) if t in self.atom_ind_group[1]]

        affected = set()

        # Neighbor range (max of B-B and B-A distances)
        max_range = 4.2  # Angstroms

        for b_idx in b_site_indices:
            dist1 = poscar['dismat'][b_idx, swapped_idx1]
            dist2 = poscar['dismat'][b_idx, swapped_idx2]

            if dist1 < max_range or dist2 < max_range:
                affected.add(b_idx)

        return affected


    def _incremental_update(self, poscar, affected_sites, structure_id='default'):
        """
        Update cluster counts incrementally after swap.

        Args:
            poscar: Updated structure
            affected_sites: Set of B-sites to recompute
            structure_id: Cache ID

        Returns:
            cluster_list: Updated counts
        """
        atom_types = create_atom_type_mapping(poscar)

        # Get cached cluster counts
        if structure_id not in self._cluster_cache:
            # No cache, do full computation
            return self._count_clusters_from_poscar(poscar, use_cache=False, structure_id=None)

        cluster_cache = self._cluster_cache[structure_id]
        cluster_counts = [0] * len(self.reference_clusters)

        # Count from cache for unaffected sites
        b_site_indices = [i for i, t in enumerate(atom_types) if t in self.atom_ind_group[1]]

        for b_idx in b_site_indices:
            if b_idx in affected_sites:
                # Recompute
                core_type = atom_types[b_idx]
                positioned_neighbors = find_positioned_neighbors(
                    b_idx, poscar, atom_types, self.atom_ind_group
                )
                cluster = generate_single_positioned_cluster(core_type, positioned_neighbors)

                if cluster is None:
                    continue

                canonical = get_canonical_form(tuple(cluster))
                cluster_cache[b_idx] = canonical
            else:
                # Use cache
                canonical = cluster_cache.get(b_idx, None)
                if canonical is None:
                    continue

            # Add to counts
            ref_idx = self.cluster_to_idx.get(canonical, None)
            if ref_idx is not None:
                cluster_counts[ref_idx] += 1

        # Prepend B-site and O-site counts (unchanged by swap within same site)
        b_count = sum(1 for t in atom_types if t in self.atom_ind_group[1])
        o_count = sum(1 for t in atom_types if t in self.atom_ind_group[2])

        result = [b_count, o_count] + cluster_counts

        return result


def create_energy_calculator(base_dir='src/energy_models/cluster_expansion/energy_parameter',
                            atom_ind_group=None, element_names=None):
    """
    Convenience function to create EnergyCalculator.

    Args:
        base_dir: Directory containing model files
        atom_ind_group: Optional custom atom groups
                       Default: [[0], [1, 2], [3, 4]] for Sr-Ti/Fe-O/VO system
        element_names: Optional element names for tensor conversion
                      Default: ['Sr', 'Ti', 'Fe', 'O', 'VO']

    Returns:
        calculator: Configured EnergyCalculator instance
    """
    import os

    model_file = os.path.join(base_dir, 'trained_lasso_model.pkl')
    scaler_file = os.path.join(base_dir, 'trained_lasso_scaler.pkl')
    cluster_file = os.path.join(base_dir, 'reference_clusters.json')
    template_file = os.path.join(base_dir, 'POSCAR_ABO3')

    # Default for Sr-Ti/Fe-O/VO substitution system
    if atom_ind_group is None:
        atom_ind_group = [
            [0],      # A-site: Sr
            [1, 2],   # B-site: Ti, Fe
            [3, 4]    # O-site: O, VO
        ]

    if element_names is None:
        element_names = ['Sr', 'Ti', 'Fe', 'O', 'VO']

    calculator = EnergyCalculator(
        model_file=model_file,
        scaler_file=scaler_file,
        cluster_file=cluster_file,
        atom_ind_group=atom_ind_group,
        element_names=element_names,
        template_file=template_file
    )

    return calculator
