/**
 * Sublattice-Constrained Swap Operations (C++)
 *
 * Header-only implementation for CPU/CUDA.
 * Can be used standalone or with LibTorch.
 *
 * For ABO3 perovskite structures:
 * - B-site: Ti ↔ Fe only
 * - O-site: O ↔ VO only
 */

#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sublattice_swap {

// =============================================================================
// Data Structures
// =============================================================================

struct SwapResult {
    std::vector<std::vector<int>> swapped;  // [batch, N]
    std::vector<std::pair<int, int>> indices;  // [batch]
};

struct BeamResult {
    std::vector<std::vector<std::vector<int>>> swapped;  // [batch, beam, N]
    std::vector<std::vector<std::pair<int, int>>> indices;  // [batch, beam]
    std::vector<std::vector<double>> log_probs;  // [batch, beam]
};

// =============================================================================
// Random Number Generation
// =============================================================================

class GumbelSampler {
public:
    explicit GumbelSampler(unsigned int seed = 42) : gen_(seed), uniform_(0.0, 1.0) {}

    double sample() {
        double u = uniform_(gen_);
        u = std::max(u, 1e-10);  // clamp
        return -std::log(-std::log(u));
    }

    std::vector<double> sample_n(size_t n) {
        std::vector<double> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = sample();
        }
        return result;
    }

private:
    std::mt19937 gen_;
    std::uniform_real_distribution<double> uniform_;
};

// =============================================================================
// Basic Operations
// =============================================================================

/**
 * Swap elements at indices idx[0] and idx[1] in vector x.
 */
inline std::vector<int> swap_by_idx(
    const std::vector<int>& x,
    int idx_a,
    int idx_b
) {
    std::vector<int> result = x;
    std::swap(result[idx_a], result[idx_b]);
    return result;
}

/**
 * Get indices where atom_types[i] == type_a or atom_types[i] == type_b
 */
inline std::vector<int> get_sublattice_indices(
    const std::vector<int>& atom_types,
    int type_a,
    int type_b
) {
    std::vector<int> indices;
    for (size_t i = 0; i < atom_types.size(); ++i) {
        if (atom_types[i] == type_a || atom_types[i] == type_b) {
            indices.push_back(static_cast<int>(i));
        }
    }
    return indices;
}

/**
 * Compute log-softmax of scores (masked version).
 * Positions where mask[i] == false get -inf.
 */
inline std::vector<double> masked_log_softmax(
    const std::vector<double>& scores,
    const std::vector<bool>& mask
) {
    double max_val = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < scores.size(); ++i) {
        if (mask[i]) {
            max_val = std::max(max_val, scores[i]);
        }
    }

    double sum_exp = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (mask[i]) {
            sum_exp += std::exp(scores[i] - max_val);
        }
    }
    double log_sum_exp = max_val + std::log(sum_exp);

    std::vector<double> result(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        if (mask[i]) {
            result[i] = scores[i] - log_sum_exp;
        } else {
            result[i] = -std::numeric_limits<double>::infinity();
        }
    }
    return result;
}

// =============================================================================
// Single Swap (Gumbel-Max)
// =============================================================================

/**
 * Sample one constrained swap for a single structure.
 *
 * @param atom_types [N] atom type indices
 * @param sublattice_indices indices of sublattice atoms
 * @param type_a, type_b types to swap (must be different)
 * @param scores [N] swap scores (empty = uniform)
 * @param sampler random sampler
 * @return pair of (swapped_types, swap_indices)
 */
inline std::pair<std::vector<int>, std::pair<int, int>> sample_sublattice_swap_single(
    const std::vector<int>& atom_types,
    const std::vector<int>& sublattice_indices,
    int type_a,
    int type_b,
    const std::vector<double>& scores,
    GumbelSampler& sampler
) {
    size_t M = sublattice_indices.size();
    bool has_scores = !scores.empty();

    // Extract sublattice
    std::vector<int> sub_types(M);
    std::vector<double> sub_scores(M, 0.0);

    for (size_t i = 0; i < M; ++i) {
        int idx = sublattice_indices[i];
        sub_types[i] = atom_types[idx];
        if (has_scores) {
            sub_scores[i] = scores[idx];
        }
    }

    // Add Gumbel noise
    auto gumbel_a = sampler.sample_n(M);
    auto gumbel_b = sampler.sample_n(M);

    // Find best type_a and type_b positions
    double best_score_a = -std::numeric_limits<double>::infinity();
    double best_score_b = -std::numeric_limits<double>::infinity();
    int best_local_a = -1;
    int best_local_b = -1;

    for (size_t i = 0; i < M; ++i) {
        double score_with_noise_a = sub_scores[i] + gumbel_a[i];
        double score_with_noise_b = sub_scores[i] + gumbel_b[i];

        if (sub_types[i] == type_a && score_with_noise_a > best_score_a) {
            best_score_a = score_with_noise_a;
            best_local_a = static_cast<int>(i);
        }
        if (sub_types[i] == type_b && score_with_noise_b > best_score_b) {
            best_score_b = score_with_noise_b;
            best_local_b = static_cast<int>(i);
        }
    }

    if (best_local_a < 0 || best_local_b < 0) {
        throw std::runtime_error("Could not find valid swap pair");
    }

    // Map to global indices
    int global_a = sublattice_indices[best_local_a];
    int global_b = sublattice_indices[best_local_b];

    // Perform swap
    auto swapped = swap_by_idx(atom_types, global_a, global_b);

    return {swapped, {global_a, global_b}};
}

/**
 * Sample constrained swap for batch of structures.
 */
inline SwapResult sample_sublattice_swap(
    const std::vector<std::vector<int>>& atom_types_batch,
    const std::vector<int>& sublattice_indices,
    int type_a,
    int type_b,
    const std::vector<std::vector<double>>& scores_batch,
    GumbelSampler& sampler
) {
    size_t batch_size = atom_types_batch.size();
    bool has_scores = !scores_batch.empty();

    SwapResult result;
    result.swapped.resize(batch_size);
    result.indices.resize(batch_size);

    for (size_t b = 0; b < batch_size; ++b) {
        std::vector<double> scores;
        if (has_scores) {
            scores = scores_batch[b];
        }

        auto [swapped, indices] = sample_sublattice_swap_single(
            atom_types_batch[b],
            sublattice_indices,
            type_a,
            type_b,
            scores,
            sampler
        );

        result.swapped[b] = std::move(swapped);
        result.indices[b] = indices;
    }

    return result;
}

// =============================================================================
// Beam Search
// =============================================================================

/**
 * Beam search for top-k swap candidates (single structure).
 */
inline std::tuple<
    std::vector<std::vector<int>>,  // [beam, N]
    std::vector<std::pair<int, int>>,  // [beam]
    std::vector<double>  // [beam]
> sample_sublattice_swap_beam_single(
    const std::vector<int>& atom_types,
    const std::vector<int>& sublattice_indices,
    int type_a,
    int type_b,
    const std::vector<double>& scores,
    int beam_size
) {
    size_t M = sublattice_indices.size();
    bool has_scores = !scores.empty();

    // Extract sublattice
    std::vector<int> sub_types(M);
    std::vector<double> sub_scores(M, 0.0);
    std::vector<int> type_a_locals, type_b_locals;

    for (size_t i = 0; i < M; ++i) {
        int idx = sublattice_indices[i];
        sub_types[i] = atom_types[idx];
        if (has_scores) {
            sub_scores[i] = scores[idx];
        }
        if (sub_types[i] == type_a) {
            type_a_locals.push_back(static_cast<int>(i));
        } else if (sub_types[i] == type_b) {
            type_b_locals.push_back(static_cast<int>(i));
        }
    }

    // Create masks for log_softmax
    std::vector<bool> mask_a(M, false), mask_b(M, false);
    for (int i : type_a_locals) mask_a[i] = true;
    for (int i : type_b_locals) mask_b[i] = true;

    // Log-softmax for probabilities
    auto log_prob_a = masked_log_softmax(sub_scores, mask_a);
    auto log_prob_b = masked_log_softmax(sub_scores, mask_b);

    // Sort type_a and type_b by score
    std::vector<std::pair<double, int>> scored_a, scored_b;
    for (int i : type_a_locals) {
        scored_a.emplace_back(sub_scores[i], i);
    }
    for (int i : type_b_locals) {
        scored_b.emplace_back(sub_scores[i], i);
    }

    std::sort(scored_a.rbegin(), scored_a.rend());
    std::sort(scored_b.rbegin(), scored_b.rend());

    int k_a = std::min(beam_size, static_cast<int>(scored_a.size()));
    int k_b = std::min(beam_size, static_cast<int>(scored_b.size()));

    // Generate all pairs and score them
    std::vector<std::tuple<double, int, int>> pair_scores;  // (score, local_a, local_b)

    for (int i = 0; i < k_a; ++i) {
        for (int j = 0; j < k_b; ++j) {
            int local_a = scored_a[i].second;
            int local_b = scored_b[j].second;
            double pair_score = sub_scores[local_a] + sub_scores[local_b];
            pair_scores.emplace_back(pair_score, local_a, local_b);
        }
    }

    // Sort and take top beam_size
    std::sort(pair_scores.rbegin(), pair_scores.rend());

    int actual_beam = std::min(beam_size, static_cast<int>(pair_scores.size()));

    std::vector<std::vector<int>> swapped_candidates;
    std::vector<std::pair<int, int>> indices_candidates;
    std::vector<double> log_probs_candidates;

    for (int b = 0; b < actual_beam; ++b) {
        auto [score, local_a, local_b] = pair_scores[b];

        int global_a = sublattice_indices[local_a];
        int global_b = sublattice_indices[local_b];

        auto swapped = swap_by_idx(atom_types, global_a, global_b);
        double log_prob = log_prob_a[local_a] + log_prob_b[local_b];

        swapped_candidates.push_back(std::move(swapped));
        indices_candidates.emplace_back(global_a, global_b);
        log_probs_candidates.push_back(log_prob);
    }

    return {swapped_candidates, indices_candidates, log_probs_candidates};
}

/**
 * Beam search for batch of structures.
 */
inline BeamResult sample_sublattice_swap_beam(
    const std::vector<std::vector<int>>& atom_types_batch,
    const std::vector<int>& sublattice_indices,
    int type_a,
    int type_b,
    const std::vector<std::vector<double>>& scores_batch,
    int beam_size
) {
    size_t batch_size = atom_types_batch.size();
    bool has_scores = !scores_batch.empty();

    BeamResult result;
    result.swapped.resize(batch_size);
    result.indices.resize(batch_size);
    result.log_probs.resize(batch_size);

    for (size_t b = 0; b < batch_size; ++b) {
        std::vector<double> scores;
        if (has_scores) {
            scores = scores_batch[b];
        } else {
            scores.resize(atom_types_batch[b].size(), 0.0);
        }

        auto [swapped, indices, log_probs] = sample_sublattice_swap_beam_single(
            atom_types_batch[b],
            sublattice_indices,
            type_a,
            type_b,
            scores,
            beam_size
        );

        result.swapped[b] = std::move(swapped);
        result.indices[b] = std::move(indices);
        result.log_probs[b] = std::move(log_probs);
    }

    return result;
}

// =============================================================================
// Log Probability
// =============================================================================

/**
 * Compute log P(swap_indices | scores) for a single structure.
 */
inline double log_prob_sublattice_swap_single(
    const std::vector<double>& scores,
    const std::vector<int>& sublattice_indices,
    int type_a,
    int type_b,
    const std::vector<int>& atom_types,
    std::pair<int, int> swap_indices
) {
    size_t M = sublattice_indices.size();

    // Extract sublattice
    std::vector<int> sub_types(M);
    std::vector<double> sub_scores(M);

    for (size_t i = 0; i < M; ++i) {
        int idx = sublattice_indices[i];
        sub_types[i] = atom_types[idx];
        sub_scores[i] = scores[idx];
    }

    // Create masks
    std::vector<bool> mask_a(M, false), mask_b(M, false);
    for (size_t i = 0; i < M; ++i) {
        if (sub_types[i] == type_a) mask_a[i] = true;
        if (sub_types[i] == type_b) mask_b[i] = true;
    }

    auto log_prob_a = masked_log_softmax(sub_scores, mask_a);
    auto log_prob_b = masked_log_softmax(sub_scores, mask_b);

    // Find local indices
    int local_a = -1, local_b = -1;
    for (size_t i = 0; i < M; ++i) {
        if (sublattice_indices[i] == swap_indices.first) local_a = static_cast<int>(i);
        if (sublattice_indices[i] == swap_indices.second) local_b = static_cast<int>(i);
    }

    return log_prob_a[local_a] + log_prob_b[local_b];
}

// =============================================================================
// Multiple Swaps
// =============================================================================

/**
 * Apply n swap steps to batch.
 */
inline std::pair<
    std::vector<std::vector<int>>,  // final [batch, N]
    std::vector<std::vector<std::pair<int, int>>>  // history [n_swaps, batch]
> apply_n_swaps(
    const std::vector<std::vector<int>>& atom_types_batch,
    const std::vector<int>& sublattice_indices,
    int type_a,
    int type_b,
    int n_swaps,
    GumbelSampler& sampler
) {
    auto current = atom_types_batch;
    std::vector<std::vector<std::pair<int, int>>> history(n_swaps);

    std::vector<std::vector<double>> empty_scores;

    for (int step = 0; step < n_swaps; ++step) {
        auto result = sample_sublattice_swap(
            current,
            sublattice_indices,
            type_a,
            type_b,
            empty_scores,
            sampler
        );
        current = std::move(result.swapped);
        history[step] = std::move(result.indices);
    }

    return {current, history};
}

}  // namespace sublattice_swap
