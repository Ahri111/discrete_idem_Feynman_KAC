"""
Optimized energy calculator with in-memory computation and optional GPU acceleration.

Key improvements:
1. No file I/O during MCMC sampling
2. Flexible atom_ind_group generation
3. Optional GPU acceleration for distance matrices
4. Batch processing optimizations
"""

import pickle
import numpy as np
from typing import List, Union, Dict, Optional
import json

from src.energy_models.cluster_expansion.structure_utils import posreader, create_atom_type_mapping
from src.energy_models.cluster_expansion.reference_generator import load_reference_clusters
from src.energy_models.cluster_expansion.inplace_calculator import (
    build_atom_ind_group,
    compute_features_from_structure_inplace
)
from src.energy_models.cluster_expansion.gpu_accelerated import (
    GPUDistanceCalculator,
    GPUEnergyPredictor,
    check_gpu_availability
)


class OptimizedEnergyCalculator:
    """
    Optimized energy calculator with minimal I/O and optional GPU acceleration.

    Improvements over basic EnergyCalculator:
    - No file I/O: compute directly from structure arrays
    - Flexible atom grouping: auto-generate atom_ind_group
    - GPU acceleration: optional CuPy/PyTorch backend
    - Efficient MCMC: reuse distance matrices when possible
    """

    def __init__(
        self,
        model_file: str,
        scaler_file: str,
        cluster_file: str,
        atom_group: Optional[List[List[str]]] = None,
        use_gpu: bool = False,
        gpu_backend: str = 'auto'
    ):
        """
        Initialize optimized energy calculator.

        Args:
            model_file: Path to trained Lasso model
            scaler_file: Path to StandardScaler
            cluster_file: Path to reference clusters JSON
            atom_group: Element grouping, e.g., [['Sr', 'La'], ['Ti'], ['O']]
                       If None, uses default perovskite grouping
            use_gpu: Enable GPU acceleration
            gpu_backend: 'cupy', 'torch', or 'auto'
        """
        # Load model and scaler
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

        with open(scaler_file, "rb") as f:
            self.scaler = pickle.load(f)

        # Load reference clusters
        self.reference_clusters = load_reference_clusters(cluster_file)

        # Atom grouping (flexible)
        self.atom_group = atom_group  # Will be converted to indices per structure

        # Default perovskite grouping if not specified
        if atom_group is None:
            self.default_atom_ind_group = [[0, 2], [1], [3]]  # A, B, O
        else:
            self.default_atom_ind_group = None

        # GPU setup
        self.use_gpu = use_gpu
        self.gpu_available = check_gpu_availability()

        if use_gpu:
            self.gpu_distance_calc = GPUDistanceCalculator(use_gpu=True)
            self.gpu_energy_pred = GPUEnergyPredictor(
                self.model, self.scaler,
                device='cuda' if self.gpu_available['torch_cuda'] else 'cpu'
            )
        else:
            self.gpu_distance_calc = None
            self.gpu_energy_pred = None

    def _get_atom_ind_group(self, poscar: Dict) -> List[List[int]]:
        """Get atom_ind_group for a structure."""
        if self.atom_group is not None:
            # Build from atom_group
            return build_atom_ind_group(self.atom_group, poscar)
        else:
            # Use default
            return self.default_atom_ind_group

    def compute_features_from_poscar(self, poscar_file: str) -> np.ndarray:
        """
        Compute features from POSCAR file.

        Args:
            poscar_file: Path to POSCAR file

        Returns:
            features: Feature vector for energy prediction
        """
        # Load structure
        poscar = posreader(poscar_file)
        atom_types = create_atom_type_mapping(poscar)
        atom_ind_group = self._get_atom_ind_group(poscar)

        # Extract arrays
        positions = np.array(poscar['LattPnt'])
        lattice = np.array(poscar['Base'])

        # Compute features
        features = compute_features_from_structure_inplace(
            positions, lattice, atom_types,
            atom_ind_group, self.reference_clusters
        )

        return features

    def compute_features_inplace(
        self,
        positions: np.ndarray,
        lattice: np.ndarray,
        atom_types: List[int],
        atom_ind_group: List[List[int]]
    ) -> np.ndarray:
        """
        Compute features directly from structure arrays (no file I/O).

        Args:
            positions: (N, 3) fractional coordinates
            lattice: (3, 3) lattice vectors
            atom_types: List of atom type indices
            atom_ind_group: [[A_types], [B_types], [O_types]]

        Returns:
            features: Feature vector for energy prediction
        """
        return compute_features_from_structure_inplace(
            positions, lattice, atom_types,
            atom_ind_group, self.reference_clusters
        )

    def compute_energy(self, poscar_file: str) -> float:
        """
        Compute energy from POSCAR file.

        Args:
            poscar_file: Path to POSCAR file

        Returns:
            energy: Formation energy (eV)
        """
        features = self.compute_features_from_poscar(poscar_file)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        energy = self.model.predict(features_scaled)[0]
        return energy

    def compute_energy_inplace(
        self,
        positions: np.ndarray,
        lattice: np.ndarray,
        atom_types: List[int],
        atom_ind_group: List[List[int]]
    ) -> float:
        """
        Compute energy directly from structure arrays (no file I/O).

        This is the optimized version for MCMC sampling.

        Args:
            positions: (N, 3) fractional coordinates
            lattice: (3, 3) lattice vectors
            atom_types: List of atom type indices
            atom_ind_group: [[A_types], [B_types], [O_types]]

        Returns:
            energy: Formation energy (eV)
        """
        features = self.compute_features_inplace(
            positions, lattice, atom_types, atom_ind_group
        )

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        energy = self.model.predict(features_scaled)[0]

        return energy

    def compute_energy_batch(
        self,
        poscar_files: List[str] = None,
        structures: List[Dict] = None,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Compute energies for multiple structures.

        Args:
            poscar_files: List of POSCAR file paths (optional)
            structures: List of structure dicts with 'positions', 'lattice',
                       'atom_types', 'atom_ind_group' (optional)
            batch_size: Batch size for GPU processing

        Returns:
            energies: Array of formation energies
        """
        if poscar_files is not None:
            # Load from files
            all_features = []
            for poscar_file in poscar_files:
                try:
                    features = self.compute_features_from_poscar(poscar_file)
                    all_features.append(features)
                except Exception as e:
                    print(f"Error processing {poscar_file}: {e}")
                    all_features.append(None)

            valid_features = [f for f in all_features if f is not None]

        elif structures is not None:
            # Compute from structure arrays
            valid_features = []
            for struct in structures:
                features = self.compute_features_inplace(
                    struct['positions'],
                    struct['lattice'],
                    struct['atom_types'],
                    struct['atom_ind_group']
                )
                valid_features.append(features)

        else:
            raise ValueError("Must provide either poscar_files or structures")

        if len(valid_features) == 0:
            return np.array([])

        # Stack features
        features_batch = np.array(valid_features)

        # Predict energies (with optional GPU acceleration)
        if self.use_gpu and self.gpu_energy_pred is not None:
            energies = self.gpu_energy_pred.predict_batch(features_batch)
        else:
            features_scaled = self.scaler.transform(features_batch)
            energies = self.model.predict(features_scaled)

        return energies

    def get_gpu_info(self) -> Dict:
        """Get GPU availability and performance info."""
        return {
            'gpu_enabled': self.use_gpu,
            'gpu_available': self.gpu_available,
            'distance_calc_gpu': self.gpu_distance_calc is not None,
            'energy_pred_gpu': self.gpu_energy_pred is not None,
        }
