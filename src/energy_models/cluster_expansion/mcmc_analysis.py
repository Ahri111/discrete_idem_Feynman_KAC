"""
MCMC Analysis and Plotting Tools

Tools for analyzing MCMC diagnostics:
- Energy trace visualization
- Autocorrelation analysis
- Acceptance rate tracking
- Effective sample size estimation
- Statistical diagnostics

Usage:
    from mcmc_analysis import analyze_mcmc_run, plot_all_diagnostics

    # Analyze diagnostics file
    analyze_mcmc_run('diagnostics.json', output_dir='plots/')

    # Or plot individual diagnostics
    plot_energy_trace(energy_trace, 'energy_trace.png')
    plot_autocorrelation(autocorr, tau, 'autocorr.png')
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os


def load_diagnostics(filename: str) -> Dict:
    """Load diagnostics from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_energy_trace(
    energy_trace: List[float],
    output_file: Optional[str] = None,
    equilibration_end: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot energy vs step.

    Args:
        energy_trace: List of energies
        output_file: Save to file (optional)
        equilibration_end: Mark equilibration end with vertical line
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    steps = np.arange(len(energy_trace))
    energies = np.array(energy_trace)

    # Full trace
    ax1.plot(steps, energies, 'b-', alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Energy (eV)', fontsize=12)
    ax1.set_title('Energy Trace', fontsize=14)
    ax1.grid(True, alpha=0.3)

    if equilibration_end:
        ax1.axvline(equilibration_end, color='r', linestyle='--',
                   label=f'Equilibration ({equilibration_end} steps)')
        ax1.legend()

    # Running average
    window = min(1000, len(energies) // 10)
    if window > 1:
        running_avg = np.convolve(energies, np.ones(window)/window, mode='valid')
        ax2.plot(steps[window-1:], running_avg, 'g-', linewidth=2,
                label=f'Running avg (window={window})')
        ax2.plot(steps, energies, 'b-', alpha=0.2, linewidth=0.5)
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Energy (eV)', fontsize=12)
        ax2.set_title('Energy Running Average', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved energy trace to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_autocorrelation(
    autocorr: np.ndarray,
    tau: float,
    output_file: Optional[str] = None,
    max_lag: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 5)
):
    """
    Plot autocorrelation function.

    Args:
        autocorr: Autocorrelation array
        tau: Integrated autocorrelation time
        output_file: Save to file (optional)
        max_lag: Maximum lag to plot
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    lags = np.arange(len(autocorr))

    if max_lag:
        lags = lags[:max_lag]
        autocorr = autocorr[:max_lag]

    # Linear scale
    ax1.plot(lags, autocorr, 'b-', linewidth=2)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.axhline(0.05, color='r', linestyle='--', alpha=0.5, label='Cutoff (0.05)')
    ax1.axvline(tau, color='g', linestyle='--', linewidth=2,
               label=f'τ = {tau:.1f}')
    ax1.set_xlabel('Lag (steps)', fontsize=12)
    ax1.set_ylabel('Autocorrelation C(t)', fontsize=12)
    ax1.set_title('Autocorrelation Function', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log scale
    # Only plot positive values
    positive_autocorr = autocorr.copy()
    positive_autocorr[positive_autocorr <= 0] = np.nan

    ax2.semilogy(lags, positive_autocorr, 'b-', linewidth=2)
    ax2.axhline(0.05, color='r', linestyle='--', alpha=0.5, label='Cutoff (0.05)')
    ax2.axvline(tau, color='g', linestyle='--', linewidth=2,
               label=f'τ = {tau:.1f}')
    ax2.set_xlabel('Lag (steps)', fontsize=12)
    ax2.set_ylabel('Autocorrelation C(t) [log]', fontsize=12)
    ax2.set_title('Autocorrelation (Log Scale)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved autocorrelation to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_acceptance_rate(
    acceptance_trace: List[bool],
    output_file: Optional[str] = None,
    window: int = 1000,
    figsize: Tuple[int, int] = (10, 5)
):
    """
    Plot acceptance rate over time.

    Args:
        acceptance_trace: List of boolean accepts
        output_file: Save to file (optional)
        window: Window size for moving average
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    accepts = np.array(acceptance_trace, dtype=float)
    steps = np.arange(len(accepts))

    # Moving average
    if len(accepts) > window:
        moving_avg = np.convolve(accepts, np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], moving_avg, 'b-', linewidth=2,
               label=f'Moving avg (window={window})')

    # Overall average
    overall_avg = np.mean(accepts)
    ax.axhline(overall_avg, color='r', linestyle='--', linewidth=2,
              label=f'Overall avg: {overall_avg:.3f}')

    # Recommended range
    ax.axhspan(0.3, 0.5, alpha=0.1, color='g', label='Recommended range')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Acceptance Rate', fontsize=12)
    ax.set_title('MCMC Acceptance Rate', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved acceptance rate to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_energy_histogram(
    energy_trace: List[float],
    output_file: Optional[str] = None,
    bins: int = 50,
    figsize: Tuple[int, int] = (8, 5)
):
    """
    Plot energy distribution histogram.

    Args:
        energy_trace: List of energies
        output_file: Save to file (optional)
        bins: Number of histogram bins
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    energies = np.array(energy_trace)

    ax.hist(energies, bins=bins, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(np.mean(energies), color='r', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(energies):.4f} eV')
    ax.axvline(np.median(energies), color='g', linestyle='--', linewidth=2,
              label=f'Median: {np.median(energies):.4f} eV')

    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Energy Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved energy histogram to {output_file}")
    else:
        plt.show()

    plt.close()


def compute_effective_sample_size(n_samples: int, tau: float) -> int:
    """
    Compute effective sample size.

    N_eff = N / (2 * tau)

    Args:
        n_samples: Number of collected samples
        tau: Autocorrelation time

    Returns:
        n_eff: Effective sample size
    """
    n_eff = int(n_samples / (2 * tau))
    return max(1, n_eff)


def analyze_equilibration(
    energy_trace: List[float],
    window: int = 1000,
    threshold: float = 0.01
) -> int:
    """
    Estimate when equilibration occurred.

    Uses running variance to detect when energy stabilizes.

    Args:
        energy_trace: Energy time series
        window: Window size for statistics
        threshold: Relative variance threshold

    Returns:
        equilibration_step: Estimated equilibration point
    """
    energies = np.array(energy_trace)

    if len(energies) < 2 * window:
        return len(energies) // 2

    # Compute running variance
    running_var = []
    for i in range(window, len(energies) - window):
        var = np.var(energies[i:i+window])
        running_var.append(var)

    running_var = np.array(running_var)

    # Find when variance stabilizes
    if len(running_var) == 0:
        return window

    var_threshold = np.median(running_var) * (1 + threshold)

    for i, var in enumerate(running_var):
        if var < var_threshold:
            return i + window

    return len(energies) // 2


def print_summary_statistics(diagnostics: Dict):
    """
    Print summary statistics from diagnostics.

    Args:
        diagnostics: Diagnostics dictionary
    """
    print("\n" + "="*60)
    print("MCMC Summary Statistics")
    print("="*60)

    # Configuration
    config = diagnostics.get('config', {})
    print(f"\nConfiguration:")
    print(f"  Temperature: {config.get('temperature', 'N/A')} K")
    print(f"  Equilibration steps: {config.get('n_equilibration', 'N/A')}")
    print(f"  Autocorr measure steps: {config.get('n_autocorr_measure', 'N/A')}")
    print(f"  Target samples: {config.get('n_samples', 'N/A')}")
    print(f"  Swap mode: {config.get('swap_mode', 'N/A')}")

    # Energy statistics
    energy_trace = np.array(diagnostics['energy_trace'])
    print(f"\nEnergy Statistics:")
    print(f"  Initial: {energy_trace[0]:.6f} eV")
    print(f"  Final: {energy_trace[-1]:.6f} eV")
    print(f"  Mean: {np.mean(energy_trace):.6f} eV")
    print(f"  Std: {np.std(energy_trace):.6f} eV")
    print(f"  Min: {np.min(energy_trace):.6f} eV")
    print(f"  Max: {np.max(energy_trace):.6f} eV")

    # Acceptance statistics
    acceptance_trace = np.array(diagnostics['acceptance_trace'])
    overall_accept = np.mean(acceptance_trace)
    print(f"\nAcceptance Statistics:")
    print(f"  Overall rate: {overall_accept:.3f}")

    # Last 1000 steps
    if len(acceptance_trace) > 1000:
        recent_accept = np.mean(acceptance_trace[-1000:])
        print(f"  Recent rate (last 1000): {recent_accept:.3f}")

    # Autocorrelation
    if 'autocorr' in diagnostics:
        tau = diagnostics['autocorr']['tau']
        print(f"\nAutocorrelation:")
        print(f"  τ: {tau:.1f} steps")
        print(f"  Recommended thinning: {int(5*tau)} steps")

        # Effective sample size
        if config.get('n_samples'):
            n_eff = compute_effective_sample_size(config['n_samples'], tau)
            print(f"  Effective samples: {n_eff}")

    print("\n" + "="*60 + "\n")


def plot_all_diagnostics(
    diagnostics: Dict,
    output_dir: str = 'plots',
    show_plots: bool = False
):
    """
    Generate all diagnostic plots.

    Args:
        diagnostics: Diagnostics dictionary
        output_dir: Output directory for plots
        show_plots: Show plots interactively (in addition to saving)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Generating diagnostic plots in {output_dir}/]")

    # Energy trace
    equilibration_end = diagnostics['config'].get('n_equilibration', None)
    plot_energy_trace(
        diagnostics['energy_trace'],
        output_file=os.path.join(output_dir, 'energy_trace.png'),
        equilibration_end=equilibration_end
    )

    # Acceptance rate
    plot_acceptance_rate(
        diagnostics['acceptance_trace'],
        output_file=os.path.join(output_dir, 'acceptance_rate.png')
    )

    # Energy histogram
    plot_energy_histogram(
        diagnostics['energy_trace'],
        output_file=os.path.join(output_dir, 'energy_histogram.png')
    )

    # Autocorrelation (if available)
    if 'autocorr' in diagnostics:
        autocorr_data = diagnostics['autocorr']
        plot_autocorrelation(
            np.array(autocorr_data['autocorr']),
            autocorr_data['tau'],
            output_file=os.path.join(output_dir, 'autocorrelation.png')
        )

    print(f"[All plots generated]")


def analyze_mcmc_run(
    diagnostics_file: str,
    output_dir: str = 'analysis',
    show_plots: bool = False
):
    """
    Complete analysis of MCMC run.

    Args:
        diagnostics_file: Path to diagnostics JSON
        output_dir: Output directory for plots and reports
        show_plots: Show plots interactively
    """
    print(f"\n{'='*60}")
    print(f"Analyzing MCMC Run: {diagnostics_file}")
    print(f"{'='*60}")

    # Load diagnostics
    diagnostics = load_diagnostics(diagnostics_file)

    # Print summary
    print_summary_statistics(diagnostics)

    # Generate plots
    plot_all_diagnostics(diagnostics, output_dir=output_dir, show_plots=show_plots)

    # Save text report
    report_file = os.path.join(output_dir, 'summary.txt')
    with open(report_file, 'w') as f:
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        print_summary_statistics(diagnostics)

        report_text = sys.stdout.getvalue()
        sys.stdout = old_stdout

        f.write(report_text)

    print(f"[Report saved to {report_file}]")
    print(f"[Analysis complete]\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mcmc_analysis.py <diagnostics.json> [output_dir]")
        sys.exit(1)

    diagnostics_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'analysis'

    analyze_mcmc_run(diagnostics_file, output_dir=output_dir)
