"""
MCMC Sampler for Diffusion Model Training Data

Standard Metropolis-Hastings sampling at fixed temperature with:
- Equilibration period
- Autocorrelation measurement
- Production sampling with thinning
- Full diagnostics

Usage:
    sampler = MCMCSampler(
        calculator=energy_calculator,
        composition={'A': {...}, 'B': {...}, 'O': {...}},
        temperature=300.0,
        atom_ind_group=[[0], [1, 2], [3, 4]]
    )

    # Phase 1: Equilibration
    sampler.equilibrate(n_steps=20000)

    # Phase 2: Measure autocorrelation
    tau = sampler.measure_autocorrelation(n_steps=5000)

    # Phase 3: Collect samples
    samples = sampler.sample(n_samples=1000, thinning='auto')
"""

import numpy as np
import copy
import random
import time
import json
import os
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict

from src.energy_models.cluster_expansion.energy_calculator import EnergyCalculator
from src.energy_models.cluster_expansion.random_structure_generator import generate_random_structure
from src.energy_models.cluster_expansion.structure_utils import dismatswap, poswriter


@dataclass
class MCMCState:
    """Current state of MCMC sampler"""
    step: int
    energy: float
    acceptance_rate: float
    temperature: float


@dataclass
class MCMCConfig:
    """
    MCMC sampling configuration

    Args:
        temperature: Temperature in Kelvin
        n_equilibration: Number of equilibration steps
        n_autocorr_measure: Steps for autocorrelation measurement
        n_samples: Target number of independent samples
        thinning: Thinning interval (None = auto-determine from tau)
        swap_mode: Which sites to swap
            'B-site': Only swap B-site pairs (e.g., Ti <-> Fe)
            'O-site': Only swap O-site pairs (e.g., O <-> VO)
            'both': Randomly choose B or O each step
        save_interval: Progress print interval (steps)
        seed: Random seed for reproducibility
    """
    temperature: float = 300.0
    n_equilibration: int = 20000
    n_autocorr_measure: int = 5000
    n_samples: int = 1000
    thinning: Optional[int] = None
    swap_mode: str = 'B-site'
    save_interval: int = 100
    seed: Optional[int] = None


class MCMCSampler:
    """
    Metropolis-Hastings MCMC sampler for perovskite structures.

    Args:
        calculator: EnergyCalculator instance
        composition: Composition dict (e.g., {'A': {'Sr': 32}, 'B': {'Ti': 24, 'Fe': 8}, ...})
        template_file: Path to template POSCAR
        atom_ind_group: [[A_types], [B_types], [O_types]]
        element_names: List of element names
        config: MCMCConfig instance (optional)
    """

    def __init__(
        self,
        calculator: EnergyCalculator,
        composition: Dict,
        template_file: str,
        atom_ind_group: List[List[int]],
        element_names: List[str],
        config: Optional[MCMCConfig] = None
    ):
        self.calculator = calculator
        self.composition = composition
        self.template_file = template_file
        self.atom_ind_group = atom_ind_group
        self.element_names = element_names

        # Configuration
        self.config = config if config else MCMCConfig()

        # Boltzmann constant (eV/K)
        self.kB = 8.617333e-5
        self.beta = 1.0 / (self.kB * self.config.temperature)

        # State
        self.current_state = None
        self.current_energy = None
        self.step = 0

        # Diagnostics
        self.energy_trace = []
        self.acceptance_trace = []

        # Site indices for swapping
        self._setup_swap_sites()

        # Random seed
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        print(f"[MCMCSampler] Initialized")
        print(f"  Temperature: {self.config.temperature} K")
        print(f"  Beta: {self.beta:.6f} eV^-1")
        print(f"  Swap mode: {self.config.swap_mode}")


    def _setup_swap_sites(self):
        """Setup atom indices for swapping based on swap_mode"""
        # Count atoms to get ranges
        a_count = sum(self.composition.get('A', {}).values())
        b_count = sum(self.composition.get('B', {}).values())
        o_count = sum(self.composition.get('O', {}).values())

        # Total site ranges
        self.a_range = (0, a_count)
        self.b_range = (a_count, a_count + b_count)
        self.o_range = (a_count + b_count, a_count + b_count + o_count)

        # Detailed ranges for each element type (for efficient swap)
        # B-site element ranges
        self.b_type_ranges = {}
        current_idx = a_count
        for elem, count in sorted(self.composition.get('B', {}).items()):
            self.b_type_ranges[elem] = (current_idx, current_idx + count)
            current_idx += count

        # O-site element ranges
        self.o_type_ranges = {}
        current_idx = a_count + b_count
        for elem, count in sorted(self.composition.get('O', {}).items()):
            self.o_type_ranges[elem] = (current_idx, current_idx + count)
            current_idx += count

        print(f"[MCMCSampler] Atom ranges:")
        print(f"  A-site: {self.a_range[0]}-{self.a_range[1]-1}")
        print(f"  B-site: {self.b_range[0]}-{self.b_range[1]-1}")
        for elem, (start, end) in self.b_type_ranges.items():
            print(f"    {elem}: {start}-{end-1}")
        print(f"  O-site: {self.o_range[0]}-{self.o_range[1]-1}")
        for elem, (start, end) in self.o_type_ranges.items():
            print(f"    {elem}: {start}-{end-1}")

        # Setup swap pairs (different types only)
        self.swap_pairs = []

        # B-site pairs (e.g., Ti <-> Fe)
        b_types = list(self.b_type_ranges.keys())
        if len(b_types) >= 2:
            # Always swap first two types (e.g., Ti <-> Fe)
            self.swap_pairs.append(('B', b_types[0], b_types[1]))

        # O-site pairs (e.g., O <-> VO)
        o_types = list(self.o_type_ranges.keys())
        if len(o_types) >= 2:
            # Always swap first two types (e.g., O <-> VO)
            self.swap_pairs.append(('O', o_types[0], o_types[1]))

        if len(self.swap_pairs) == 0:
            raise ValueError("No swap pairs available! Need at least 2 types in B-site or O-site")

        print(f"[MCMCSampler] Swap pairs:")
        for site, type1, type2 in self.swap_pairs:
            print(f"  {site}-site: {type1} <-> {type2}")


    def initialize(self, initial_state=None):
        """
        Initialize MCMC with random or provided state.

        Args:
            initial_state: Optional POSCAR dict to start from
        """
        if initial_state is None:
            print("[MCMCSampler] Generating random initial state...")
            self.current_state = generate_random_structure(
                template_file=self.template_file,
                composition=self.composition,
                element_names=self.element_names
            )
        else:
            self.current_state = copy.deepcopy(initial_state)

        # Compute initial energy
        print("[MCMCSampler] Computing initial energy...")
        self.current_energy = self.calculator.compute_energy(
            self.current_state,
            use_cache=True,
            structure_id='mcmc'
        )

        print(f"[MCMCSampler] Initial energy: {self.current_energy:.6f} eV")

        self.step = 0
        self.energy_trace = [self.current_energy]
        self.acceptance_trace = []


    def metropolis_step(self) -> bool:
        """
        Perform one Metropolis-Hastings step with efficient swap.

        Always swaps different types (e.g., Ti <-> Fe, O <-> VO) for efficiency.
        Based on original algorithm but using incremental updates.

        Returns:
            accepted: Whether the move was accepted
        """
        # Select swap pair based on mode
        if self.config.swap_mode == 'B-site':
            # Only B-site pairs
            available_pairs = [p for p in self.swap_pairs if p[0] == 'B']
        elif self.config.swap_mode == 'O-site':
            # Only O-site pairs
            available_pairs = [p for p in self.swap_pairs if p[0] == 'O']
        elif self.config.swap_mode == 'both':
            # All pairs
            available_pairs = self.swap_pairs
        else:
            raise ValueError(f"Unknown swap_mode: {self.config.swap_mode}")

        if len(available_pairs) == 0:
            return False

        # Randomly select one pair
        site, type1, type2 = random.choice(available_pairs)

        # Get ranges for each type
        if site == 'B':
            range1 = self.b_type_ranges[type1]
            range2 = self.b_type_ranges[type2]
        else:  # site == 'O'
            range1 = self.o_type_ranges[type1]
            range2 = self.o_type_ranges[type2]

        # Select one atom from each type
        idx1 = random.randint(range1[0], range1[1] - 1)
        idx2 = random.randint(range2[0], range2[1] - 1)

        # Swap atoms (ALWAYS different types)
        self.current_state['LattPnt'][idx1], self.current_state['LattPnt'][idx2] = \
            self.current_state['LattPnt'][idx2], self.current_state['LattPnt'][idx1]

        # Update distance matrix
        self.current_state['dismat'] = dismatswap(
            self.current_state['dismat'], idx1, idx2
        )

        # Compute new energy (incremental)
        energy_new = self.calculator.compute_energy_incremental(
            self.current_state, idx1, idx2, structure_id='mcmc'
        )

        # Metropolis criterion
        delta_E = energy_new - self.current_energy

        if delta_E <= 0:
            # Accept
            self.current_energy = energy_new
            return True
        else:
            # Accept with probability exp(-β ΔE)
            prob = np.exp(-self.beta * delta_E)
            if random.random() < prob:
                # Accept
                self.current_energy = energy_new
                return True
            else:
                # Reject - swap back
                self.current_state['LattPnt'][idx1], self.current_state['LattPnt'][idx2] = \
                    self.current_state['LattPnt'][idx2], self.current_state['LattPnt'][idx1]
                self.current_state['dismat'] = dismatswap(
                    self.current_state['dismat'], idx1, idx2
                )
                return False


    def equilibrate(self, n_steps: Optional[int] = None, verbose: bool = True):
        """
        Run equilibration phase.

        Args:
            n_steps: Number of equilibration steps (default: config.n_equilibration)
            verbose: Print progress
        """
        if n_steps is None:
            n_steps = self.config.n_equilibration

        print(f"\n{'='*60}")
        print(f"Phase 1: Equilibration ({n_steps} steps)")
        print(f"{'='*60}")

        acceptance_count = 0
        start_time = time.time()

        for i in range(n_steps):
            accepted = self.metropolis_step()

            if accepted:
                acceptance_count += 1

            self.energy_trace.append(self.current_energy)
            self.acceptance_trace.append(accepted)
            self.step += 1

            # Progress
            if verbose and i % self.config.save_interval == 0:
                accept_rate = acceptance_count / (i + 1)
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed

                print(f"  Step {i:6d}/{n_steps} | "
                      f"Energy: {self.current_energy:8.4f} eV | "
                      f"Accept: {accept_rate:.3f} | "
                      f"Rate: {rate:.1f} steps/s")

        final_accept_rate = acceptance_count / n_steps
        elapsed = time.time() - start_time

        print(f"\n[Equilibration Complete]")
        print(f"  Final energy: {self.current_energy:.6f} eV")
        print(f"  Final acceptance rate: {final_accept_rate:.3f}")
        print(f"  Time: {elapsed:.2f} s ({n_steps/elapsed:.1f} steps/s)")

        return final_accept_rate


    def measure_autocorrelation(
        self,
        n_steps: Optional[int] = None,
        max_lag: int = 1000,
        verbose: bool = True
    ) -> float:
        """
        Measure autocorrelation time.

        Args:
            n_steps: Number of steps to measure (default: config.n_autocorr_measure)
            max_lag: Maximum lag for autocorrelation
            verbose: Print results

        Returns:
            tau: Integrated autocorrelation time
        """
        if n_steps is None:
            n_steps = self.config.n_autocorr_measure

        print(f"\n{'='*60}")
        print(f"Phase 2: Autocorrelation Measurement ({n_steps} steps)")
        print(f"{'='*60}")

        energies = []
        start_time = time.time()

        for i in range(n_steps):
            self.metropolis_step()
            energies.append(self.current_energy)
            self.step += 1

            if verbose and i % 1000 == 0:
                print(f"  Step {i:6d}/{n_steps}")

        # Compute autocorrelation
        energies = np.array(energies)
        autocorr = self._compute_autocorrelation(energies, max_lag)
        tau = self._integrated_autocorr_time(autocorr)

        elapsed = time.time() - start_time

        print(f"\n[Autocorrelation Measurement Complete]")
        print(f"  Autocorrelation time τ: {tau:.1f} steps")
        print(f"  Recommended thinning: {int(5 * tau)} steps")
        print(f"  Time: {elapsed:.2f} s")

        # Store autocorrelation data
        self.autocorr_data = {
            'autocorr': autocorr,
            'tau': tau,
            'energies': energies
        }

        return tau


    def sample(
        self,
        n_samples: Optional[int] = None,
        thinning: Optional[int] = None,
        save_dir: Optional[str] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Production sampling with thinning.

        Args:
            n_samples: Number of independent samples (default: config.n_samples)
            thinning: Thinning interval (default: 5*tau or config.thinning)
            save_dir: Directory to save samples (optional)
            verbose: Print progress

        Returns:
            samples: List of POSCAR dicts
        """
        if n_samples is None:
            n_samples = self.config.n_samples

        # Determine thinning
        if thinning is None:
            if self.config.thinning is not None:
                thinning = self.config.thinning
            elif hasattr(self, 'autocorr_data'):
                # Use 5*tau
                thinning = max(int(5 * self.autocorr_data['tau']), 10)
            else:
                raise ValueError("Thinning not specified and autocorrelation not measured")

        total_steps = n_samples * thinning

        print(f"\n{'='*60}")
        print(f"Phase 3: Production Sampling")
        print(f"{'='*60}")
        print(f"  Target samples: {n_samples}")
        print(f"  Thinning interval: {thinning}")
        print(f"  Total steps: {total_steps}")

        samples = []
        acceptance_count = 0
        start_time = time.time()

        for i in range(total_steps):
            accepted = self.metropolis_step()

            if accepted:
                acceptance_count += 1

            self.step += 1

            # Save sample
            if i % thinning == 0:
                samples.append(copy.deepcopy(self.current_state))

                if verbose:
                    accept_rate = acceptance_count / (i + 1)
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed

                    print(f"  Sample {len(samples):4d}/{n_samples} | "
                          f"Step {i:6d}/{total_steps} | "
                          f"Energy: {self.current_energy:8.4f} eV | "
                          f"Accept: {accept_rate:.3f} | "
                          f"Rate: {rate:.1f} steps/s")

        elapsed = time.time() - start_time
        final_accept_rate = acceptance_count / total_steps

        print(f"\n[Production Sampling Complete]")
        print(f"  Collected samples: {len(samples)}")
        print(f"  Final acceptance rate: {final_accept_rate:.3f}")
        print(f"  Time: {elapsed:.2f} s ({total_steps/elapsed:.1f} steps/s)")

        # Save samples if requested
        if save_dir is not None:
            self._save_samples(samples, save_dir)

        return samples


    def _compute_autocorrelation(self, x: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute autocorrelation function"""
        mean = np.mean(x)
        var = np.var(x)

        if var == 0:
            return np.ones(max_lag)

        corr = []
        for lag in range(max_lag):
            if len(x) - lag < 10:
                break
            c = np.mean((x[:-lag or None] - mean) * (x[lag:] - mean)) / var
            corr.append(c)

        return np.array(corr)


    def _integrated_autocorr_time(self, autocorr: np.ndarray, cutoff: float = 0.05) -> float:
        """Compute integrated autocorrelation time"""
        tau = 1.0
        for i in range(1, len(autocorr)):
            if autocorr[i] < cutoff:
                break
            tau += 2 * autocorr[i]
        return tau


    def _save_samples(self, samples: List[Dict], save_dir: str):
        """Save samples to directory"""
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n[Saving {len(samples)} samples to {save_dir}]")

        for i, sample in enumerate(samples):
            filename = os.path.join(save_dir, f'sample_{i:05d}.vasp')
            poswriter(filename, sample)

        # Save metadata
        metadata = {
            'n_samples': len(samples),
            'thinning': self.config.thinning,
            'temperature': self.config.temperature,
            'tau': self.autocorr_data.get('tau', None) if hasattr(self, 'autocorr_data') else None,
            'composition': self.composition,
            'element_names': self.element_names
        }

        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved to {save_dir}")


    def save_diagnostics(self, filename: str):
        """Save full diagnostics (energy trace, acceptance, etc.)"""
        diagnostics = {
            'config': asdict(self.config),
            'energy_trace': self.energy_trace,
            'acceptance_trace': [int(a) for a in self.acceptance_trace],
            'total_steps': self.step,
            'final_energy': self.current_energy
        }

        if hasattr(self, 'autocorr_data'):
            diagnostics['autocorr'] = {
                'tau': self.autocorr_data['tau'],
                'autocorr': self.autocorr_data['autocorr'].tolist()
            }

        with open(filename, 'w') as f:
            json.dump(diagnostics, f, indent=2)

        print(f"[Diagnostics saved to {filename}]")
