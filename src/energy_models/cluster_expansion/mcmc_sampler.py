import numpy as np
import copy
import random
import math
import pickle
import json
import time
import os
from typing import Optional, List, Dict, Tuple

from src.energy_models.cluster_expansion.structure_utils import (
    posreader, poswriter, dismatcreate, dismatswap, create_atom_type_mapping
)
from src.energy_models.cluster_expansion.cluster_counter import count_cluster
from src.energy_models.cluster_expansion.energy_calculator import EnergyCalculator


class MCMCSampler:
    """
    Metropolis-Hastings MCMC sampler for perovskite structures.

    Performs Monte Carlo sampling with atom swaps and energy evaluation
    using cluster expansion.
    """

    def __init__(
        self,
        energy_calculator: EnergyCalculator,
        temperature: float = 1000.0,
        swap_types: Optional[List[Tuple[int, int]]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize MCMC sampler.

        Args:
            energy_calculator: EnergyCalculator instance for energy evaluation
            temperature: Temperature in Kelvin for Metropolis criterion
            swap_types: List of (type1, type2) tuples defining allowed swaps
                       If None, allows all A-site swaps: [(0, 2)]
            random_seed: Random seed for reproducibility
        """
        self.calculator = energy_calculator
        self.temperature = temperature
        self.kb = 8.617333262145e-5  # Boltzmann constant in eV/K

        # Default: allow swapping between A-site atoms (types 0 and 2)
        self.swap_types = swap_types if swap_types is not None else [(0, 2)]

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Statistics tracking
        self.reset_statistics()

    def reset_statistics(self):
        """Reset sampling statistics."""
        self.n_accepted = 0
        self.n_rejected = 0
        self.energy_history = []

    def get_acceptance_rate(self):
        """Get current acceptance rate."""
        total = self.n_accepted + self.n_rejected
        return self.n_accepted / total if total > 0 else 0.0

    def find_swappable_pairs(self, poscar, atom_types):
        """
        Find all pairs of atoms that can be swapped.

        Args:
            poscar: Structure dictionary
            atom_types: List of atom type indices

        Returns:
            swap_pairs: List of (idx1, idx2) tuples for swappable atom pairs
        """
        swap_pairs = []

        for type1, type2 in self.swap_types:
            # Find all atoms of each type
            indices_type1 = [i for i, t in enumerate(atom_types) if t == type1]
            indices_type2 = [i for i, t in enumerate(atom_types) if t == type2]

            # Create all pairs
            for i1 in indices_type1:
                for i2 in indices_type2:
                    swap_pairs.append((i1, i2))

        return swap_pairs

    def propose_swap(self, poscar, atom_types):
        """
        Propose a random atom swap.

        Args:
            poscar: Structure dictionary
            atom_types: List of atom type indices

        Returns:
            swap_pair: (idx1, idx2) tuple, or None if no swaps possible
        """
        swap_pairs = self.find_swappable_pairs(poscar, atom_types)

        if not swap_pairs:
            return None

        return random.choice(swap_pairs)

    def apply_swap(self, poscar, atom_types, idx1, idx2):
        """
        Apply atom swap to structure.

        Args:
            poscar: Structure dictionary (modified in-place)
            atom_types: List of atom types (modified in-place)
            idx1: First atom index
            idx2: Second atom index
        """
        # Swap positions
        poscar['LattPnt'][idx1], poscar['LattPnt'][idx2] = \
            poscar['LattPnt'][idx2], poscar['LattPnt'][idx1]

        # Swap atom types
        atom_types[idx1], atom_types[idx2] = atom_types[idx2], atom_types[idx1]

        # Update distance matrix efficiently
        if 'dismat' in poscar:
            poscar['dismat'] = dismatswap(poscar['dismat'], idx1, idx2)

    def metropolis_criterion(self, current_energy, proposed_energy):
        """
        Evaluate Metropolis acceptance criterion.

        Args:
            current_energy: Energy of current state
            proposed_energy: Energy of proposed state

        Returns:
            accept: True if proposal should be accepted
        """
        delta_e = proposed_energy - current_energy

        # Always accept if energy decreases
        if delta_e <= 0:
            return True

        # Accept with probability exp(-ΔE/kT) if energy increases
        beta = 1.0 / (self.kb * self.temperature)
        acceptance_prob = math.exp(-beta * delta_e)

        return random.random() < acceptance_prob

    def compute_energy_from_poscar(self, poscar, temp_file='temp_poscar.vasp'):
        """
        Compute energy for a structure.

        Args:
            poscar: Structure dictionary
            temp_file: Temporary file to write structure

        Returns:
            energy: Formation energy in eV
        """
        # Write structure to temporary file
        poswriter(temp_file, poscar)

        # Compute energy
        energy = self.calculator.compute_energy(temp_file)

        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)

        return energy

    def run_single_step(self, poscar, atom_types, current_energy):
        """
        Perform a single MCMC step.

        Args:
            poscar: Current structure (may be modified)
            atom_types: Current atom types (may be modified)
            current_energy: Current energy

        Returns:
            new_energy: Energy after step
            accepted: Whether proposal was accepted
        """
        # Propose swap
        swap_pair = self.propose_swap(poscar, atom_types)

        if swap_pair is None:
            # No valid swaps possible
            return current_energy, False

        idx1, idx2 = swap_pair

        # Save current state
        saved_poscar = copy.deepcopy(poscar)
        saved_atom_types = atom_types.copy()

        # Apply swap
        self.apply_swap(poscar, atom_types, idx1, idx2)

        # Compute new energy
        proposed_energy = self.compute_energy_from_poscar(poscar)

        if proposed_energy is None:
            # Energy calculation failed, reject
            poscar.update(saved_poscar)
            atom_types[:] = saved_atom_types
            self.n_rejected += 1
            return current_energy, False

        # Metropolis criterion
        if self.metropolis_criterion(current_energy, proposed_energy):
            # Accept
            self.n_accepted += 1
            return proposed_energy, True
        else:
            # Reject - restore previous state
            poscar.update(saved_poscar)
            atom_types[:] = saved_atom_types
            self.n_rejected += 1
            return current_energy, False

    def run(
        self,
        initial_poscar_file: str,
        n_steps: int,
        output_dir: Optional[str] = None,
        save_interval: int = 100,
        verbose: bool = True
    ):
        """
        Run MCMC sampling.

        Args:
            initial_poscar_file: Path to initial structure
            n_steps: Number of MCMC steps
            output_dir: Directory to save trajectory (if None, don't save)
            save_interval: Save structure every N steps
            verbose: Print progress information

        Returns:
            trajectory: List of (energy, poscar, atom_types) tuples
        """
        # Load initial structure
        poscar = posreader(initial_poscar_file)
        poscar = dismatcreate(poscar)
        atom_types = create_atom_type_mapping(poscar)

        # Compute initial energy
        current_energy = self.compute_energy_from_poscar(poscar)

        if current_energy is None:
            raise ValueError("Failed to compute initial energy")

        # Reset statistics
        self.reset_statistics()

        # Trajectory storage
        trajectory = [(current_energy, copy.deepcopy(poscar), atom_types.copy())]
        self.energy_history = [current_energy]

        # Create output directory
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # Run MCMC
        for step in range(n_steps):
            # Perform step
            current_energy, accepted = self.run_single_step(
                poscar, atom_types, current_energy
            )

            # Record
            self.energy_history.append(current_energy)

            # Save trajectory
            if (step + 1) % save_interval == 0:
                trajectory.append((current_energy, copy.deepcopy(poscar), atom_types.copy()))

                if output_dir is not None:
                    output_file = os.path.join(output_dir, f'structure_{step+1:06d}.vasp')
                    poswriter(output_file, poscar)

            # Print progress
            if verbose and (step + 1) % 100 == 0:
                acc_rate = self.get_acceptance_rate()
                print(f"Step {step+1}/{n_steps} | "
                      f"Energy: {current_energy:.6f} eV | "
                      f"Acceptance: {acc_rate:.3f}")

        if verbose:
            print(f"\nMCMC completed:")
            print(f"  Final energy: {current_energy:.6f} eV")
            print(f"  Overall acceptance rate: {self.get_acceptance_rate():.3f}")

        return trajectory

    def run_batch(
        self,
        initial_poscar_files: List[str],
        n_steps: int,
        output_dirs: Optional[List[str]] = None,
        save_interval: int = 100,
        verbose: bool = True
    ):
        """
        Run MCMC sampling for multiple initial structures.

        Args:
            initial_poscar_files: List of initial structure paths
            n_steps: Number of MCMC steps per structure
            output_dirs: List of output directories (or None)
            save_interval: Save structure every N steps
            verbose: Print progress information

        Returns:
            trajectories: List of trajectories, one per input structure
        """
        if output_dirs is None:
            output_dirs = [None] * len(initial_poscar_files)

        trajectories = []

        for i, (poscar_file, output_dir) in enumerate(zip(initial_poscar_files, output_dirs)):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Running MCMC for structure {i+1}/{len(initial_poscar_files)}")
                print(f"{'='*60}")

            trajectory = self.run(
                poscar_file,
                n_steps,
                output_dir=output_dir,
                save_interval=save_interval,
                verbose=verbose
            )

            trajectories.append(trajectory)

        return trajectories
