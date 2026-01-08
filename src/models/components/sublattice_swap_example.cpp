/**
 * Example usage of sublattice_swap.hpp
 *
 * Compile:
 *   g++ -std=c++17 -O3 sublattice_swap_example.cpp -o swap_example
 *
 * Run:
 *   ./swap_example
 */

#include <iostream>
#include <iomanip>
#include "sublattice_swap.hpp"

using namespace sublattice_swap;

void print_vector(const std::vector<int>& v, const std::string& name) {
    std::cout << name << ": [";
    for (size_t i = 0; i < std::min(v.size(), size_t(20)); ++i) {
        std::cout << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    if (v.size() > 20) std::cout << "...";
    std::cout << "]" << std::endl;
}

int main() {
    std::cout << "=== C++ Sublattice Swap Demo ===" << std::endl << std::endl;

    // Create structure: Sr=0(32), Ti=1(16), Fe=2(16), O=3(88), VO=4(8)
    std::vector<int> atom_types;
    for (int i = 0; i < 32; ++i) atom_types.push_back(0);  // Sr
    for (int i = 0; i < 16; ++i) atom_types.push_back(1);  // Ti
    for (int i = 0; i < 16; ++i) atom_types.push_back(2);  // Fe
    for (int i = 0; i < 88; ++i) atom_types.push_back(3);  // O
    for (int i = 0; i < 8; ++i) atom_types.push_back(4);   // VO

    std::cout << "Structure: " << atom_types.size() << " atoms" << std::endl;
    std::cout << "  Sr=32, Ti=16, Fe=16, O=88, VO=8" << std::endl;

    // Get B-site indices (Ti + Fe)
    auto b_site_indices = get_sublattice_indices(atom_types, 1, 2);
    std::cout << "\nB-site indices: " << b_site_indices.size() << " atoms" << std::endl;

    // Create batch
    int batch_size = 4;
    std::vector<std::vector<int>> batch(batch_size, atom_types);
    std::cout << "Batch size: " << batch_size << std::endl;

    // Initialize random sampler
    GumbelSampler sampler(42);

    // =========================================================================
    // 1. Single Swap
    // =========================================================================
    std::cout << "\n--- 1. Single Swap (Gumbel-Max) ---" << std::endl;

    std::vector<std::vector<double>> empty_scores;
    auto result = sample_sublattice_swap(
        batch, b_site_indices, 1, 2, empty_scores, sampler
    );

    std::cout << "Swap indices per batch:" << std::endl;
    for (int b = 0; b < batch_size; ++b) {
        auto [idx_a, idx_b] = result.indices[b];
        std::cout << "  Batch " << b << ": (" << idx_a << ", " << idx_b << ")"
                  << " types: " << atom_types[idx_a] << " <-> " << atom_types[idx_b]
                  << std::endl;
    }

    // =========================================================================
    // 2. Beam Search
    // =========================================================================
    std::cout << "\n--- 2. Beam Search (k=4) ---" << std::endl;

    // Create random scores
    std::vector<std::vector<double>> scores_batch(batch_size);
    std::mt19937 gen(123);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int b = 0; b < batch_size; ++b) {
        scores_batch[b].resize(atom_types.size());
        for (auto& s : scores_batch[b]) s = dist(gen);
    }

    auto beam_result = sample_sublattice_swap_beam(
        batch, b_site_indices, 1, 2, scores_batch, 4
    );

    std::cout << "Batch 0 beam candidates:" << std::endl;
    for (size_t k = 0; k < beam_result.indices[0].size(); ++k) {
        auto [idx_a, idx_b] = beam_result.indices[0][k];
        double log_prob = beam_result.log_probs[0][k];
        std::cout << "  Candidate " << k << ": (" << idx_a << ", " << idx_b << ")"
                  << " log_prob=" << std::fixed << std::setprecision(4) << log_prob
                  << std::endl;
    }

    // =========================================================================
    // 3. Multiple Swaps
    // =========================================================================
    std::cout << "\n--- 3. Multiple Swaps (n=10) ---" << std::endl;

    auto [final_batch, history] = apply_n_swaps(
        batch, b_site_indices, 1, 2, 10, sampler
    );

    std::cout << "Swap history (batch 0):" << std::endl;
    for (int step = 0; step < 5; ++step) {
        auto [idx_a, idx_b] = history[step][0];
        std::cout << "  Step " << step + 1 << ": (" << idx_a << ", " << idx_b << ")" << std::endl;
    }
    std::cout << "  ... and 5 more" << std::endl;

    // =========================================================================
    // 4. Verify Composition
    // =========================================================================
    std::cout << "\n--- 4. Composition Verification ---" << std::endl;

    bool all_ok = true;
    for (int b = 0; b < batch_size; ++b) {
        int count_ti = 0, count_fe = 0;
        for (int t : final_batch[b]) {
            if (t == 1) count_ti++;
            if (t == 2) count_fe++;
        }
        if (count_ti != 16 || count_fe != 16) {
            std::cout << "Batch " << b << ": FAILED (Ti=" << count_ti << ", Fe=" << count_fe << ")" << std::endl;
            all_ok = false;
        }
    }
    if (all_ok) {
        std::cout << "All batches preserve composition (Ti=16, Fe=16)" << std::endl;
    }

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
