"""
MCMC Sampling Example for Diffusion Training Data

Complete workflow:
1. Initialize energy calculator
2. Setup MCMC sampler
3. Phase 1: Equilibration
4. Phase 2: Measure autocorrelation
5. Phase 3: Collect samples
6. Analyze results
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.energy_models.cluster_expansion.energy_calculator import create_energy_calculator
from src.energy_models.cluster_expansion.mcmc_sampler import MCMCSampler, MCMCConfig
from src.energy_models.cluster_expansion.mcmc_analysis import analyze_mcmc_run


def example_mcmc_sampling():
    """Complete MCMC sampling workflow"""

    print("\n" + "="*60)
    print("MCMC Sampling for Diffusion Training Data")
    print("="*60)

    # ============================================
    # 1. Setup
    # ============================================
    print("\n[1] Setting up energy calculator...")

    base_dir = 'src/energy_models/cluster_expansion/energy_parameter'
    template_file = os.path.join(base_dir, 'POSCAR_ABO3')

    # Create energy calculator
    calculator = create_energy_calculator(
        base_dir=base_dir,
        atom_ind_group=[[0], [1, 2], [3, 4]],  # Sr, Ti/Fe, O/VO
        element_names=['Sr', 'Ti', 'Fe', 'O', 'VO']
    )

    # Composition (adjust as needed)
    composition = {
        'A': {'Sr': 32},
        'B': {'Ti': 24, 'Fe': 8},    # 75% Ti, 25% Fe
        'O': {'O': 88, 'VO': 4}       # ~4% vacancy
    }

    # MCMC configuration (ADJUST THESE!)
    config = MCMCConfig(
        temperature=300.0,           # K (adjust for different datasets)
        n_equilibration=10000,       # Equilibration steps (20000 recommended)
        n_autocorr_measure=5000,     # Autocorrelation measurement steps
        n_samples=100,               # Target independent samples (1000+ for real)
        thinning=None,               # Auto-determine from tau
        swap_mode='B-site',          # 'B-site', 'O-site', or 'both'
        save_interval=100,           # Progress print interval
        seed=42                      # Random seed (for reproducibility)
    )

    # ============================================
    # 2. Initialize MCMC
    # ============================================
    print("\n[2] Initializing MCMC sampler...")

    sampler = MCMCSampler(
        calculator=calculator,
        composition=composition,
        template_file=template_file,
        atom_ind_group=[[0], [1, 2], [3, 4]],
        element_names=['Sr', 'Ti', 'Fe', 'O', 'VO'],
        config=config
    )

    # Initialize with random structure
    sampler.initialize()

    # ============================================
    # 3. Phase 1: Equilibration
    # ============================================
    print("\n[3] Running equilibration...")

    acceptance_rate = sampler.equilibrate()

    if acceptance_rate < 0.2:
        print("\n⚠️  WARNING: Acceptance rate too low!")
        print("   Consider increasing temperature or adjusting swap strategy")
    elif acceptance_rate > 0.7:
        print("\n⚠️  WARNING: Acceptance rate too high!")
        print("   Consider decreasing temperature")
    else:
        print("\n✓ Acceptance rate in good range (0.2-0.7)")

    # ============================================
    # 4. Phase 2: Measure Autocorrelation
    # ============================================
    print("\n[4] Measuring autocorrelation...")

    tau = sampler.measure_autocorrelation()

    print(f"\n✓ Measured τ = {tau:.1f} steps")
    print(f"  Thinning will be: {int(5*tau)} steps")

    # ============================================
    # 5. Phase 3: Production Sampling
    # ============================================
    print("\n[5] Collecting samples...")

    output_dir = 'mcmc_samples'
    samples = sampler.sample(
        n_samples=config.n_samples,
        save_dir=output_dir
    )

    print(f"\n✓ Collected {len(samples)} samples")

    # ============================================
    # 6. Save Diagnostics
    # ============================================
    print("\n[6] Saving diagnostics...")

    diagnostics_file = 'mcmc_diagnostics.json'
    sampler.save_diagnostics(diagnostics_file)

    print(f"✓ Diagnostics saved to {diagnostics_file}")

    # ============================================
    # 7. Analyze Results
    # ============================================
    print("\n[7] Analyzing results...")

    analyze_mcmc_run(
        diagnostics_file=diagnostics_file,
        output_dir='mcmc_analysis'
    )

    print("\n" + "="*60)
    print("MCMC Sampling Complete!")
    print("="*60)
    print(f"\nResults:")
    print(f"  Samples: {len(samples)} independent structures")
    print(f"  Saved to: {output_dir}/")
    print(f"  Diagnostics: {diagnostics_file}")
    print(f"  Analysis plots: mcmc_analysis/")
    print("\n" + "="*60 + "\n")


def example_custom_config():
    """Example with custom configuration"""

    print("\n" + "="*60)
    print("Custom MCMC Configuration Example")
    print("="*60)

    # High temperature for diverse sampling
    config_high_T = MCMCConfig(
        temperature=1000.0,          # High T → more diversity
        n_equilibration=50000,       # Longer equilibration
        n_autocorr_measure=10000,
        n_samples=5000,              # Large dataset
        swap_mode='both',            # Swap both B and O sites
        seed=None                    # Random seed each run
    )

    # Low temperature for low-energy structures
    config_low_T = MCMCConfig(
        temperature=100.0,           # Low T → near ground state
        n_equilibration=100000,      # Much longer (slow mixing)
        n_autocorr_measure=20000,
        n_samples=1000,
        swap_mode='B-site',
        seed=123
    )

    print("\nHigh-T config:")
    print(f"  T = {config_high_T.temperature} K")
    print(f"  Purpose: Diverse sampling for diffusion training")

    print("\nLow-T config:")
    print(f"  T = {config_low_T.temperature} K")
    print(f"  Purpose: Low-energy structures near ground state")

    print("\nTo use:")
    print("  sampler = MCMCSampler(..., config=config_high_T)")


def example_different_compositions():
    """Example with different compositions"""

    print("\n" + "="*60)
    print("Different Composition Examples")
    print("="*60)

    # Pure SrTiO3 (no substitution)
    comp_pure = {
        'A': {'Sr': 32},
        'B': {'Ti': 32},
        'O': {'O': 96}
    }

    # 50% Fe substitution
    comp_50Fe = {
        'A': {'Sr': 32},
        'B': {'Ti': 16, 'Fe': 16},
        'O': {'O': 92, 'VO': 4}
    }

    # High vacancy
    comp_high_vac = {
        'A': {'Sr': 32},
        'B': {'Ti': 24, 'Fe': 8},
        'O': {'O': 84, 'VO': 12}  # 12.5% vacancy
    }

    comps = [
        ("Pure SrTiO3", comp_pure),
        ("50% Fe", comp_50Fe),
        ("High Vacancy", comp_high_vac)
    ]

    for name, comp in comps:
        print(f"\n{name}:")
        for site, elements in comp.items():
            total = sum(elements.values())
            print(f"  {site}-site ({total} total):")
            for elem, count in elements.items():
                pct = count / total * 100
                print(f"    {elem}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    # Run main example
    example_mcmc_sampling()

    # Show other examples
    print("\n\n")
    example_custom_config()

    print("\n\n")
    example_different_compositions()
