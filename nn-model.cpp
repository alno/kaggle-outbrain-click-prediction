#include "nn-model.h"
#include "util/model-helpers.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>


constexpr ffm_ulong n_fields = 40;
constexpr ffm_ulong n_features = 1 << ffm_hash_bits;

constexpr uint l0_output_size = aligned_float_array_size(1 + n_fields);
constexpr uint l1_output_size = aligned_float_array_size(64);
constexpr uint l2_output_size = aligned_float_array_size(64);

constexpr uint l1_layer_size = l0_output_size * (l1_output_size - 1);
constexpr uint l2_layer_size = l1_output_size * (l2_output_size - 1);
constexpr uint l3_layer_size = l2_output_size;


class state_buffer {
public:
    float * l0_output;
    float * l0_output_grad;

    float * l1_output;
    float * l1_output_grad;
    float * l1_dropout_mask;

    float * l2_output;
    float * l2_output_grad;
    float * l2_dropout_mask;
public:
    state_buffer() {
        l0_output = malloc_aligned<float>(l0_output_size);
        l0_output_grad = malloc_aligned<float>(l0_output_size);

        l1_output = malloc_aligned<float>(l1_output_size);
        l1_output_grad = malloc_aligned<float>(l1_output_size);
        l1_dropout_mask = malloc_aligned<float>(l1_output_size);

        l2_output = malloc_aligned<float>(l2_output_size);
        l2_output_grad = malloc_aligned<float>(l2_output_size);
        l2_dropout_mask = malloc_aligned<float>(l2_output_size);
    }

    ~state_buffer() {
        free(l0_output);
        free(l0_output_grad);

        free(l1_output);
        free(l1_output_grad);
        free(l1_dropout_mask);

        free(l2_output);
        free(l2_output_grad);
        free(l2_dropout_mask);
    }

};

static thread_local state_buffer local_state_buffer;


nn_model::nn_model(int seed, float eta, float lambda) {
    this->eta = eta;
    this->lambda = lambda;

    std::default_random_engine rnd(seed);

    lin_w = malloc_aligned<float>(n_features);
    lin_wg = malloc_aligned<float>(n_features);

    l1_w = malloc_aligned<float>(l1_layer_size);
    l1_wg = malloc_aligned<float>(l1_layer_size);

    l2_w = malloc_aligned<float>(l2_layer_size);
    l2_wg = malloc_aligned<float>(l2_layer_size);

    l3_w = malloc_aligned<float>(l3_layer_size);
    l3_wg = malloc_aligned<float>(l3_layer_size);

    fill_with_rand(lin_w, n_features, std::uniform_real_distribution<float>(-1.0, 1.0), rnd);
    fill_with_ones(lin_wg, n_features);

    fill_with_rand(l1_w, l1_layer_size, std::uniform_real_distribution<float>(-1.0/l1_output_size, 1.0/l1_output_size), rnd);
    fill_with_ones(l1_wg, l1_layer_size);

    fill_with_rand(l2_w, l2_layer_size, std::uniform_real_distribution<float>(-1.0, 1.0), rnd);
    fill_with_ones(l2_wg, l2_layer_size);

    fill_with_rand(l3_w, l3_layer_size, std::uniform_real_distribution<float>(-1.0, 1.0), rnd);
    fill_with_ones(l3_wg, l3_layer_size);
}


nn_model::~nn_model() {
    free(lin_w);
    free(lin_wg);

    free(l1_w);
    free(l1_wg);

    free(l2_w);
    free(l2_wg);
}


uint nn_model::get_dropout_mask_size(const ffm_feature * start, const ffm_feature * end) {
    return l1_output_size + l2_output_size;
}


float nn_model::predict(const ffm_feature * start, const ffm_feature * end, float norm, uint64_t * dropout_mask, float dropout_mult) {
    float linear_norm = end - start;
    float * l0_output = local_state_buffer.l0_output;

    float * l1_output = local_state_buffer.l1_output;
    float * l1_dropout_mask = local_state_buffer.l1_dropout_mask;

    float * l2_output = local_state_buffer.l2_output;
    float * l2_dropout_mask = local_state_buffer.l2_dropout_mask;

    uint dropout_idx = 0;

    // Prepare dropout masks

    l1_dropout_mask[0] = 1.0; // No dropout on bias
    for (uint j = 1; j < l1_output_size; ++ j)
        l1_dropout_mask[j] = test_mask_bit(dropout_mask, dropout_idx ++);

    l2_dropout_mask[0] = 1.0; // No dropout on bias
    for (uint j = 1; j < l2_output_size; ++ j)
        l2_dropout_mask[j] = test_mask_bit(dropout_mask, dropout_idx ++);

    // Compute activations

    fill_with_zero(l0_output, l0_output_size);
    fill_with_zero(l1_output, l1_output_size);
    fill_with_zero(l2_output, l2_output_size);

    l0_output[0] = 1.0; // Layer 0 bias
    l1_output[0] = 1.0; // Layer 1 bias
    l2_output[0] = 1.0; // Layer 2 bias

    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        uint index_a = fa->index &  ffm_hash_mask;
        uint field_a = fa->index >> ffm_hash_bits;
        float value_a = fa->value;

        l0_output[1 + field_a] += lin_w[index_a] * value_a / linear_norm;
    }

    // Layer 1 forward pass
    for (uint j = 1; j < l1_output_size; ++ j) {
        __m256 ymm_total = _mm256_set1_ps(0);

        for (uint i = 0; i < l0_output_size; i += 8) {
            ymm_total += _mm256_load_ps(l0_output + i) * _mm256_load_ps(l1_w + (j - 1) * l0_output_size + i);
        }

        l1_output[j] = relu(sum(ymm_total)) * l1_dropout_mask[j] * dropout_mult;
    }

    // Layer 2 forward pass
    for (uint j = 1; j < l2_output_size; ++ j) {
        __m256 ymm_total = _mm256_set1_ps(0);

        for (uint i = 0; i < l1_output_size; i += 8) {
            ymm_total += _mm256_load_ps(l1_output + i) * _mm256_load_ps(l2_w + (j - 1) * l1_output_size + i);
        }

        float l2_out = relu(sum(ymm_total)) * l2_dropout_mask[j] * dropout_mult;

        l2_output[j] = l2_out;
    }

    // Layer 3 forward pass
    float total = 0;
    for (uint i = 0; i < l2_output_size; ++ i)
        total += l2_output[i] * l3_w[i];

    return total;
}

void nn_model::update(const ffm_feature * start, const ffm_feature * end, float norm, float kappa, uint64_t * dropout_mask, float dropout_mult) {
    float linear_norm = end - start;

    float * l0_output = local_state_buffer.l0_output;
    float * l0_output_grad = local_state_buffer.l0_output_grad;

    float * l1_output = local_state_buffer.l1_output;
    float * l1_output_grad = local_state_buffer.l1_output_grad;
    float * l1_dropout_mask = local_state_buffer.l1_dropout_mask;

    float * l2_output = local_state_buffer.l2_output;
    float * l2_output_grad = local_state_buffer.l2_output_grad;
    float * l2_dropout_mask = local_state_buffer.l2_dropout_mask;

    fill_with_zero(l0_output_grad, l0_output_size);
    fill_with_zero(l1_output_grad, l1_output_size);
    fill_with_zero(l2_output_grad, l2_output_size);

    __m256 ymm_eta = _mm256_set1_ps(eta);
    __m256 ymm_lambda = _mm256_set1_ps(lambda);
    __m256 ymm_grad = _mm256_set1_ps(kappa);

    // Backprop layer 3
    for (uint i = 0; i < l2_output_size; i += 8) {
        __m256 ymm_w = _mm256_load_ps(l3_w + i);
        __m256 ymm_dm = _mm256_load_ps(l2_dropout_mask + i);

        __m256 ymm_g = (ymm_lambda * ymm_w + ymm_grad * _mm256_load_ps(l2_output + i)) * ymm_dm; // No gradient for dropped neuron
        __m256 ymm_wg = _mm256_load_ps(l3_wg + i) + ymm_g * ymm_g;

        _mm256_store_ps(l2_output_grad + i, ymm_grad * ymm_w);

        _mm256_store_ps(l3_w + i, ymm_w - ymm_eta * ymm_g * _mm256_rsqrt_ps(ymm_wg));
        _mm256_store_ps(l3_wg + i, ymm_wg);
    }

    // Backprop layer 1
    for (uint j = 1; j < l2_output_size; ++ j) {
        __m256 ymm_l2_grad = _mm256_set1_ps(l2_output_grad[j]);

        for (uint i = 0; i < l1_output_size; i += 8) {
            uint ofs = (j - 1) * l1_output_size + i;

            __m256 ymm_w = _mm256_load_ps(l2_w + ofs);
            __m256 ymm_dm = _mm256_load_ps(l1_dropout_mask + i);

            __m256 ymm_g = (ymm_lambda * ymm_w + ymm_l2_grad * _mm256_load_ps(l1_output + i)) * ymm_dm; // No gradient for dropped neuron
            __m256 ymm_wg = _mm256_load_ps(l2_wg + ofs) + ymm_g * ymm_g;

            _mm256_store_ps(l1_output_grad + i, ymm_l2_grad * ymm_w + _mm256_load_ps(l1_output_grad + i));

            _mm256_store_ps(l2_w + ofs, ymm_w - ymm_eta * ymm_g * _mm256_rsqrt_ps(ymm_wg));
            _mm256_store_ps(l2_wg + ofs, ymm_wg);
        }
    }

    // Backprop layer 1
    for (uint j = 1; j < l1_output_size; ++ j) {
        __m256 ymm_l1_grad = _mm256_set1_ps(l1_output_grad[j]);

        for (uint i = 0; i < l0_output_size; i += 8) {
            uint ofs = (j - 1) * l0_output_size + i;

            __m256 ymm_w = _mm256_load_ps(l1_w + ofs);

            __m256 ymm_g = ymm_lambda * ymm_w + ymm_l1_grad * _mm256_load_ps(l0_output + i);
            __m256 ymm_wg = _mm256_load_ps(l1_wg + ofs) + ymm_g * ymm_g;

            _mm256_store_ps(l0_output_grad + i, ymm_l1_grad * ymm_w + _mm256_load_ps(l0_output_grad + i));

            _mm256_store_ps(l1_w + ofs, ymm_w - ymm_eta * ymm_g * _mm256_rsqrt_ps(ymm_wg));
            _mm256_store_ps(l1_wg + ofs, ymm_wg);
        }
    }

    // Update linear and interaction weights
    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        uint index_a = fa->index &  ffm_hash_mask;
        uint field_a = fa->index >> ffm_hash_bits;
        float value_a = fa->value;

        float g = lambda * lin_w[index_a] + l0_output_grad[1 + field_a] * value_a / linear_norm;
        float wg = lin_wg[index_a] + g*g;

        lin_w[index_a] -= eta * g / sqrt(wg);
        lin_wg[index_a] = wg;
    }
}
