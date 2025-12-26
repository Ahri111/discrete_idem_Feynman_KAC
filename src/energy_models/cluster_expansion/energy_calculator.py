import pickle
import numpy as np
from typing import List, Union
from src.energy_models.cluster_expansion.cluster_counter import count_cluster


class EnergyCalculator:
    """
    Cluster-Expansion based Energy Calculator with batch processing support.

    Uses a pre-trained Lasso model to predict formation energies based on
    cluster counts in perovskite structures.
    """

    def __init__(self, model_file, scaler_file, cluster_file):
        """
        Initialize energy calculator with trained model and parameters.

        Args:
            model_file: Path to pickled Lasso model
            scaler_file: Path to pickled StandardScaler
            cluster_file: Path to reference clusters JSON
        """
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

        with open(scaler_file, "rb") as f:
            self.scaler = pickle.load(f)

        self.cluster_file = cluster_file

        # Atom index groups for perovskite ABO3 structure
        # [A-sites], [B-sites], [O-sites]
        self.atom_ind_group = [[0, 2], [1], [3]]

    def compute_cluster_features(self, poscar_file):
        """
        Compute cluster count features for a single structure.

        Args:
            poscar_file: Path to POSCAR file

        Returns:
            features: 1D numpy array of cluster counts, or None if failed
        """
        try:
            counts, _ = count_cluster(
                poscar_file,
                self.atom_ind_group,
                self.cluster_file,
                verbose=False
            )

            if counts is None:
                return None

            return np.array(counts)
        except Exception as e:
            print(f"Error computing clusters for {poscar_file}: {e}")
            return None

    def compute_energy(self, poscar_file):
        """
        Compute formation energy for a single structure.

        Args:
            poscar_file: Path to POSCAR file

        Returns:
            energy: Formation energy in eV, or None if computation failed
        """
        features = self.compute_cluster_features(poscar_file)

        if features is None:
            return None

        # Scale features and predict
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        energy = self.model.predict(features_scaled)[0]

        return energy

    def compute_energy_batch(self, poscar_files: List[str], batch_size: int = 32):
        """
        Compute formation energies for multiple structures in batches.

        Args:
            poscar_files: List of paths to POSCAR files
            batch_size: Number of structures to process at once

        Returns:
            energies: List of formation energies (eV), None for failed structures
        """
        energies = []

        # Process in batches
        for i in range(0, len(poscar_files), batch_size):
            batch_files = poscar_files[i:i+batch_size]
            batch_features = []
            valid_indices = []

            # Compute features for batch
            for j, poscar_file in enumerate(batch_files):
                features = self.compute_cluster_features(poscar_file)
                if features is not None:
                    batch_features.append(features)
                    valid_indices.append(j)

            # Predict energies for valid structures
            if batch_features:
                batch_features = np.array(batch_features)
                batch_features_scaled = self.scaler.transform(batch_features)
                batch_energies = self.model.predict(batch_features_scaled)

                # Fill in results (None for failed structures)
                batch_results = [None] * len(batch_files)
                for idx, energy in zip(valid_indices, batch_energies):
                    batch_results[idx] = energy

                energies.extend(batch_results)
            else:
                # All structures in batch failed
                energies.extend([None] * len(batch_files))

        return energies

    def compute_energy_from_features(self, features: np.ndarray):
        """
        Compute energy directly from cluster count features.

        Useful when features are already computed or modified.

        Args:
            features: Cluster count features (1D or 2D array)
                     Shape: (n_features,) or (batch_size, n_features)

        Returns:
            energy: Formation energy in eV (scalar or array)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        features_scaled = self.scaler.transform(features)
        energies = self.model.predict(features_scaled)

        return energies if len(energies) > 1 else energies[0]
