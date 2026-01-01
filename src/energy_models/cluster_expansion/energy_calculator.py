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


    def _compute_dismat_numpy(self, positions_np, lattice_np):
        """
        Compute distance matrix from numpy arrays (no POSCAR dict).

        Args:
            positions_np: [N, 3] numpy array of fractional coordinates
            lattice_np: [3, 3] numpy array of lattice vectors

        Returns:
            dismat: [N, N] numpy array of distances
        """
        N = positions_np.shape[0]
        dismat = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                # Fractional coordinate difference
                delta = positions_np[i] - positions_np[j]

                # Apply minimum image convention (PBC)
                delta = np.where(delta > 0.5, delta - 1, delta)
                delta = np.where(delta <= -0.5, delta + 1, delta)
                delta = np.abs(delta)

                # Convert to Cartesian and compute distance
                cart_delta = np.dot(delta, lattice_np)
                dismat[i, j] = np.linalg.norm(cart_delta)

        return dismat


    def _find_positioned_neighbors_numpy(self, core_idx, positions_np, dismat, atom_types, atom_ind_group):
        """
        Find positioned neighbors from numpy arrays (no POSCAR dict).

        Args:
            core_idx: Index of core B-site atom
            positions_np: [N, 3] numpy array of fractional coordinates
            dismat: [N, N] numpy array of distances
            atom_types: List/array of atom type indices
            atom_ind_group: [[A_types], [B_types], [O_types]]

        Returns:
            positioned_neighbors: Dict with 'b_positions', 'o_positions', 'a_positions'
        """
        distances = dismat[core_idx]
        core_pos = positions_np[core_idx]

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

            # Direction vector with PBC
            vec = positions_np[i] - core_pos
            vec = np.where(vec > 0.5, vec - 1, vec)
            vec = np.where(vec <= -0.5, vec + 1, vec)

            # Classify by type and distance
            if atom_type in atom_ind_group[1] and b_range[0] <= dist <= b_range[1]:
                candidates['b'].append((atom_type, vec))
            elif atom_type in atom_ind_group[2] and o_range[0] <= dist <= o_range[1]:
                candidates['o'].append((atom_type, vec))
            elif len(atom_ind_group) > 0 and atom_type in atom_ind_group[0] and a_range[0] <= dist <= a_range[1]:
                candidates['a'].append((atom_type, vec))

        # Import assignment functions
        from src.energy_models.cluster_expansion.cluster_counter import assign_b_o_direction, assign_a_direction

        # Assign to directional positions
        for cand in candidates['b'][:6]:
            assign_b_o_direction(cand, positioned_neighbors['b_positions'])

        for cand in candidates['o'][:6]:
            assign_b_o_direction(cand, positioned_neighbors['o_positions'])

        for cand in candidates['a'][:8]:
            assign_a_direction(cand, positioned_neighbors['a_positions'])

        return positioned_neighbors


    def _count_clusters_from_arrays(self, atom_types, positions_np, dismat,
                                     use_cache=False, structure_id=None):
        """
        Count clusters from numpy arrays (no POSCAR dict).

        Args:
            atom_types: [N] list/array of atom type indices
            positions_np: [N, 3] numpy array of fractional coordinates
            dismat: [N, N] numpy array of distances
            use_cache: Use cached clusters if available
            structure_id: Cache identifier (for diffusion denoising sequence)

        Returns:
            cluster_list: [B_count, O_count, cluster1_count, ..., clusterN_count]
        """
        # Convert to list if array
        if isinstance(atom_types, np.ndarray):
            atom_types = atom_types.tolist()

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
                positioned_neighbors = self._find_positioned_neighbors_numpy(
                    b_idx, positions_np, dismat, atom_types, self.atom_ind_group
                )
                cluster = generate_single_positioned_cluster(core_type, positioned_neighbors)

                if cluster is None:
                    continue

                canonical = get_canonical_form(tuple(cluster))

                # Cache it
                if cluster_cache is not None:
                    cluster_cache[b_idx] = canonical

            # Lookup in reference (O(1))
            ref_idx = self.cluster_to_idx.get(canonical, None)
            if ref_idx is not None:
                cluster_counts[ref_idx] += 1

        # Prepend B-site and O-site total counts
        b_count = sum(1 for t in atom_types if t in self.atom_ind_group[1])
        o_count = sum(1 for t in atom_types if t in self.atom_ind_group[2])

        result = [b_count, o_count] + cluster_counts

        return result


    def compute_energy_from_tensor(self, positions, atom_types, lattice, metadata=None,
                                    use_cache=False, structure_id=None):
        """
        Compute energy from PyTorch tensors (for diffusion model).

        Optimized: Tensor → numpy → dismat → cluster → energy
        (No POSCAR dict conversion)

        Args:
            positions: [N, 3] tensor of fractional coordinates
            atom_types: [N] tensor of type indices
            lattice: [3, 3] tensor of lattice vectors
            metadata: Optional metadata dict (unused but kept for compatibility)
            use_cache: Use cached clusters (useful for diffusion denoising sequence)
            structure_id: Cache identifier (e.g., 'diffusion_sample_0')

        Returns:
            energy: Predicted energy (float)
        """
        # Convert tensors to numpy
        positions_np = positions.cpu().numpy() if hasattr(positions, 'cpu') else np.array(positions)
        atom_types_np = atom_types.cpu().numpy() if hasattr(atom_types, 'cpu') else np.array(atom_types)
        lattice_np = lattice.cpu().numpy() if hasattr(lattice, 'cpu') else np.array(lattice)

        # Compute distance matrix
        dismat = self._compute_dismat_numpy(positions_np, lattice_np)

        # Count clusters (with optional cache)
        cluster_list = self._count_clusters_from_arrays(
            atom_types_np, positions_np, dismat,
            use_cache=use_cache, structure_id=structure_id
        )

        # Scale and predict
        cluster_scaled = self.scaler.transform([cluster_list])[0]
        energy = self.model.predict([cluster_scaled])[0]

        return energy


    def compute_energy_batch_from_tensor(self, positions, atom_types, lattice,
                                         metadata=None, n_workers=8):
        """
        Compute energies from batched tensors (for diffusion model).

        Optimized: Tensor → numpy → dismat → cluster → energy
        (No POSCAR dict conversion)

        Args:
            positions: [B, N, 3] tensor
            atom_types: [B, N] tensor
            lattice: [3, 3] or [B, 3, 3] tensor
            metadata: Optional list of metadata dicts (unused)
            n_workers: Number of parallel workers

        Returns:
            energies: [B] numpy array of energies
        """
        # Convert tensors to numpy
        positions_np = positions.cpu().numpy() if hasattr(positions, 'cpu') else np.array(positions)
        atom_types_np = atom_types.cpu().numpy() if hasattr(atom_types, 'cpu') else np.array(atom_types)
        lattice_np = lattice.cpu().numpy() if hasattr(lattice, 'cpu') else np.array(lattice)

        batch_size = positions_np.shape[0]

        # Handle single lattice for all structures
        if lattice_np.ndim == 2:
            lattice_np = np.expand_dims(lattice_np, axis=0).repeat(batch_size, axis=0)

        # Parallel processing
        def process_single(idx):
            pos = positions_np[idx]
            types = atom_types_np[idx]
            lat = lattice_np[idx]

            # Compute distance matrix
            dismat = self._compute_dismat_numpy(pos, lat)

            # Count clusters
            cluster_list = self._count_clusters_from_arrays(types, pos, dismat)

            # Scale and predict
            cluster_scaled = self.scaler.transform([cluster_list])[0]
            energy = self.model.predict([cluster_scaled])[0]

            return energy

        # Parallel compute
        with Pool(n_workers) as pool:
            energies = pool.map(process_single, range(batch_size))

        return np.array(energies)


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
