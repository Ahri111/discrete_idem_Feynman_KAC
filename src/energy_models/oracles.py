from abc import ABC, abstractmethod
import torch
import numpy as np
from src.energy_models.cluster_expansion.energy_calculator import EnergyCalculator


class EnergyOracle(ABC):
    """Abstract base class for energy oracles."""

    @abstractmethod
    def compute_energy(self, x):
        """Compute energy for input structure(s)."""
        pass

    @abstractmethod
    def to(self, device):
        """Move oracle to specified device."""
        pass


class ClusterExpansionOracle(EnergyOracle):
    """
    Cluster Expansion Energy Oracle with batch support.

    Uses Lasso-based cluster expansion to compute formation energies
    for perovskite structures.
    """

    def __init__(self, model_file, scaler_file, cluster_file, device='cpu'):
        """
        Initialize cluster expansion oracle.

        Args:
            model_file: Path to trained Lasso model
            scaler_file: Path to feature scaler
            cluster_file: Path to reference clusters
            device: Device for tensor operations (cpu/cuda)
        """
        self.calculator = EnergyCalculator(model_file, scaler_file, cluster_file)
        self.device = device

    def compute_energy(self, x, batch_size=32):
        """
        Compute formation energies for structure(s).

        Args:
            x: Input structure(s). Can be:
               - String: Path to single POSCAR file
               - List[str]: Paths to multiple POSCAR files
               - torch.Tensor: Pre-computed cluster features
               - np.ndarray: Pre-computed cluster features

        Returns:
            energies: Tensor of formation energies (eV)
        """
        # Handle different input types
        if isinstance(x, str):
            # Single POSCAR file
            energy = self.calculator.compute_energy(x)
            if energy is None:
                raise ValueError(f"Failed to compute energy for {x}")
            return torch.tensor([energy], device=self.device, dtype=torch.float32)

        elif isinstance(x, list) and all(isinstance(item, str) for item in x):
            # List of POSCAR files
            energies = self.calculator.compute_energy_batch(x, batch_size=batch_size)
            # Filter out None values and convert to tensor
            valid_energies = [e for e in energies if e is not None]
            if len(valid_energies) == 0:
                raise ValueError("Failed to compute energies for all structures")
            return torch.tensor(valid_energies, device=self.device, dtype=torch.float32)

        elif isinstance(x, torch.Tensor):
            # Pre-computed features as tensor
            features = x.cpu().numpy()
            energies = self.calculator.compute_energy_from_features(features)
            return torch.tensor(energies, device=self.device, dtype=torch.float32)

        elif isinstance(x, np.ndarray):
            # Pre-computed features as numpy array
            energies = self.calculator.compute_energy_from_features(x)
            return torch.tensor(energies, device=self.device, dtype=torch.float32)

        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

    def compute_energy_single(self, poscar_file):
        """
        Compute energy for a single structure.

        Args:
            poscar_file: Path to POSCAR file

        Returns:
            energy: Formation energy (eV) as scalar tensor
        """
        energy = self.calculator.compute_energy(poscar_file)
        if energy is None:
            raise ValueError(f"Failed to compute energy for {poscar_file}")
        return torch.tensor(energy, device=self.device, dtype=torch.float32)

    def compute_energy_batch(self, poscar_files, batch_size=32):
        """
        Compute energies for multiple structures.

        Args:
            poscar_files: List of POSCAR file paths
            batch_size: Batch size for processing

        Returns:
            energies: Tensor of formation energies
        """
        energies = self.calculator.compute_energy_batch(poscar_files, batch_size)
        valid_energies = [e for e in energies if e is not None]
        if len(valid_energies) == 0:
            raise ValueError("Failed to compute energies for all structures")
        return torch.tensor(valid_energies, device=self.device, dtype=torch.float32)

    def to(self, device):
        """
        Move oracle to specified device.

        Args:
            device: Target device (cpu/cuda)

        Returns:
            self
        """
        self.device = device
        return self

    def __call__(self, x, batch_size=32):
        """
        Callable interface for energy computation.

        Args:
            x: Structure(s) to evaluate
            batch_size: Batch size for processing

        Returns:
            energies: Formation energy tensor
        """
        return self.compute_energy(x, batch_size=batch_size)
