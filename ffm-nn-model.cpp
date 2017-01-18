#include "ffm-nn-model.h"

#include "util/model-helpers.h"
#include "util/nn-helpers.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>


constexpr ffm_ulong n_fields = 40;
constexpr ffm_ulong n_features = 1 << ffm_hash_bits;

constexpr ffm_ulong n_dim = 16;
constexpr ffm_ulong n_dim_aligned = aligned_float_array_size(n_dim);

constexpr ffm_ulong index_stride = n_fields * n_dim_aligned * 2;
constexpr ffm_ulong field_stride = n_dim_aligned * 2;

constexpr uint prefetch_depth = 1;


constexpr uint interaction_output_size = 50;

constexpr uint l0_output_size = n_dim_aligned;
constexpr uint l1_output_size = aligned_float_array_size(24);

constexpr uint l1_layer_size = l0_output_size * (l1_output_size - 1);
constexpr uint l2_layer_size = l1_output_size;


class state_buffer {
public:
    float * l0_output;
    float * l0_output_grad;
    float * l0_dropout_mask;

    float * l1_output;
    float * l1_output_grad;
    float * l1_dropout_mask;

    std::default_random_engine gen;
public:
    state_buffer() {
        l0_output = malloc_aligned<float>(l0_output_size);
        l0_output_grad = malloc_aligned<float>(l0_output_size);
        l0_dropout_mask = malloc_aligned<float>(l0_output_size);

        l1_output = malloc_aligned<float>(l1_output_size);
        l1_output_grad = malloc_aligned<float>(l1_output_size);
        l1_dropout_mask = malloc_aligned<float>(l1_output_size);
    }

    ~state_buffer() {
        free(l0_output);
        free(l0_output_grad);
        free(l0_dropout_mask);

        free(l1_output);
        free(l1_output_grad);
        free(l1_dropout_mask);
    }

};

static thread_local state_buffer local_state_buffer;

inline void prefetch_interaction_weights(float * addr) {
    for (uint i = 0, sz = field_stride * sizeof(float); i < sz; i += 64)
        _mm_prefetch(((char *)addr) + i, _MM_HINT_T1);
}


template <typename D>
static void init_interaction_weights(ffm_float * weights, ffm_uint n, D gen, std::default_random_engine & rnd) {
    ffm_float * w = weights;

    for(ffm_uint i = 0; i < n; i++) {
        for (ffm_uint d = 0; d < n_dim; d++, w++)
            *w = gen(rnd);

        for (ffm_uint d = n_dim; d < n_dim_aligned; d++, w++)
            *w = 0;

        for (ffm_uint d = n_dim_aligned; d < 2*n_dim_aligned; d++, w++)
            *w = 1;
    }
}


ffm_nn_model::ffm_nn_model(int seed, bool restricted, float eta, float ffm_lambda, float nn_lambda) {
    this->eta = eta;
    this->ffm_lambda = ffm_lambda;
    this->nn_lambda = nn_lambda;

    if (restricted) {
        max_b_field = 29;
        min_a_field = 10;
    } else {
        max_b_field = n_fields;
        min_a_field = 0;
    }

    std::default_random_engine rnd(seed);

    ffm_weights = malloc_aligned<float>(n_features * n_fields * n_dim_aligned * 2);
    lin_weights = malloc_aligned<float>(n_features * n_dim_aligned * 2);

    l1_w = malloc_aligned<float>(l1_layer_size);
    l1_wg = malloc_aligned<float>(l1_layer_size);

    l2_w = malloc_aligned<float>(l2_layer_size);
    l2_wg = malloc_aligned<float>(l2_layer_size);

    init_interaction_weights(ffm_weights, n_features * n_fields, std::uniform_real_distribution<float>(-1.0/sqrt(n_dim), 1.0/sqrt(n_dim)), rnd);
    init_interaction_weights(lin_weights, n_features, std::uniform_real_distribution<float>(-0.001, 0.001), rnd);

    fill_with_rand(l1_w, l1_layer_size, std::uniform_real_distribution<float>(-1.0/l1_output_size, 1.0/l1_output_size), rnd);
    fill_with_ones(l1_wg, l1_layer_size);

    fill_with_rand(l2_w, l2_layer_size, std::uniform_real_distribution<float>(-1.0, 1.0), rnd);
    fill_with_ones(l2_wg, l2_layer_size);
}


ffm_nn_model::~ffm_nn_model() {
    free(ffm_weights);
    free(lin_weights);

    free(l1_w);
    free(l1_wg);

    free(l2_w);
    free(l2_wg);
}


uint ffm_nn_model::get_dropout_mask_size(const ffm_feature * start, const ffm_feature * end) {
    uint feature_count = end - start;
    uint interaction_count = feature_count * (feature_count + 1) / 2;

    return interaction_count;
}


float ffm_nn_model::predict(const ffm_feature * start, const ffm_feature * end, float norm, uint64_t * dropout_mask, float dropout_mult) {
    float linear_norm = end - start;

    float * l0_output = local_state_buffer.l0_output;
    float * l0_dropout_mask = local_state_buffer.l0_dropout_mask;

    float * l1_output = local_state_buffer.l1_output;
    float * l1_dropout_mask = local_state_buffer.l1_dropout_mask;

    auto & gen = local_state_buffer.gen;

    fill_with_zero(l0_output, l0_output_size);
    fill_with_zero(l1_output, l1_output_size);

    uint dropout_idx = 0;
    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        uint index_a = fa->index &  ffm_hash_mask;
        uint field_a = fa->index >> ffm_hash_bits;
        float value_a = fa->value;

        {
            float * wl = lin_weights + index_a * field_stride;

            __m256 ymm_val = _mm256_set1_ps(value_a / linear_norm);
            for(uint d = 0; d < n_dim; d += 8) {
                _mm256_store_ps(l0_output + d,  _mm256_load_ps(l0_output + d) + _mm256_load_ps(wl + d) * ymm_val);
            }
        }

        if (field_a < min_a_field)
            continue;

        for (const ffm_feature * fb = start; fb != fa; ++ fb) {
            uint index_b = fb->index &  ffm_hash_mask;
            uint field_b = fb->index >> ffm_hash_bits;
            float value_b = fb->value;

            if (field_b > max_b_field)
                break;

            if (fb + prefetch_depth < fa && test_mask_bit(dropout_mask, dropout_idx + prefetch_depth)) { // Prefetch row only if no dropout
                uint index_p = fb[prefetch_depth].index &  ffm_hash_mask;
                uint field_p = fb[prefetch_depth].index >> ffm_hash_bits;

                prefetch_interaction_weights(ffm_weights + index_p * index_stride + field_a * field_stride);
                prefetch_interaction_weights(ffm_weights + index_a * index_stride + field_p * field_stride);
            }

            if (test_mask_bit(dropout_mask, dropout_idx ++) == 0)
                continue;

            float * wa = ffm_weights + index_a * index_stride + field_b * field_stride;
            float * wb = ffm_weights + index_b * index_stride + field_a * field_stride;

            __m256 ymm_val = _mm256_set1_ps(dropout_mult * value_a * value_b / norm);

            for(uint d = 0; d < n_dim; d += 8) {
                __m256 ymm_wa = _mm256_load_ps(wa + d);
                __m256 ymm_wb = _mm256_load_ps(wb + d);

                _mm256_store_ps(l0_output + d,  _mm256_load_ps(l0_output + d) + ymm_wa * ymm_wb * ymm_val);
            }
        }
    }

    // Prepare dropout masks
    if (dropout_mult > 1) {
        float l0_dropout_prob = 0;
        float l1_dropout_prob = 0;

        float l0_dropout_scale = 1 / (1 - l0_dropout_prob);
        float l1_dropout_scale = 1 / (1 - l1_dropout_prob);

        std::uniform_real_distribution<float> dropout_distr(0, 1);

        l0_dropout_mask[0] = 1.0; // No dropout on bias
        for (uint j = 1; j < l0_output_size; ++ j)
            l0_dropout_mask[j] = (dropout_distr(gen) >= l0_dropout_prob) * l0_dropout_scale;

        l1_dropout_mask[0] = 1.0; // No dropout on bias
        for (uint j = 1; j < l1_output_size; ++ j)
            l1_dropout_mask[j] = (dropout_distr(gen) >= l1_dropout_prob) * l1_dropout_scale;
    } else {
        fill_with_ones(l0_dropout_mask, l0_output_size);
        fill_with_ones(l1_dropout_mask, l1_output_size);
    }

    // Layer 0 relu
    l0_output[0] = 1.0; // Layer 1 bias
    for (uint j = 1; j < l0_output_size; ++ j)
        l0_output[j] = relu(l0_output[j]) * l0_dropout_mask[j];

    // Layer 1 forward pass
    l1_output[0] = 1.0; // Layer 2 bias
    for (uint j = 1; j < l1_output_size; ++ j)
        l1_output[j] = relu(forward_pass(l0_output_size, l0_output, l1_w + (j - 1) * l0_output_size)) * l1_dropout_mask[j];

    // Layer 2 forward pass
    return forward_pass(l1_output_size, l1_output, l2_w);
}


void ffm_nn_model::update(const ffm_feature * start, const ffm_feature * end, float norm, float kappa, uint64_t * dropout_mask, float dropout_mult) {
    float linear_norm = end - start;

    float * l0_output = local_state_buffer.l0_output;
    float * l0_output_grad = local_state_buffer.l0_output_grad;
    float * l0_dropout_mask = local_state_buffer.l0_dropout_mask;

    float * l1_output = local_state_buffer.l1_output;
    float * l1_output_grad = local_state_buffer.l1_output_grad;
    float * l1_dropout_mask = local_state_buffer.l1_dropout_mask;

    fill_with_zero(l0_output_grad, l0_output_size);
    fill_with_zero(l1_output_grad, l1_output_size);

    // Backprop layer 2
    backward_pass(l1_output_size, l1_output, l1_output_grad, l2_w, l2_wg, kappa, eta, nn_lambda);

    // Backprop layer 1
    for (uint j = 1, ofs = 0; j < l1_output_size; ++ j, ofs += l0_output_size) {
        float l1_grad = l1_output_grad[j] * l1_dropout_mask[j];

        if (l1_output[j] <= 0) // Relu activation: grad in negative part is zero
            l1_grad = 0;

        backward_pass(l0_output_size, l0_output, l0_output_grad, l1_w + ofs, l1_wg + ofs, l1_grad, eta, nn_lambda);
    }

    // Backprop layer 0
    l0_output_grad[0] = 0;
    for (uint j = 1; j < l0_output_size; ++ j) {
        float l0_grad = l0_output_grad[j] * l0_dropout_mask[j];

        if (l0_output[j] <= 0) // Relu activation: grad in negative part is zero
            l0_grad = 0;

        l0_output_grad[j] = l0_grad;
    }

    __m256 ymm_eta = _mm256_set1_ps(eta);
    __m256 ymm_ffm_lambda = _mm256_set1_ps(ffm_lambda);

    // Update linear and interaction weights
    uint dropout_idx = 0;
    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        uint index_a = fa->index &  ffm_hash_mask;
        uint field_a = fa->index >> ffm_hash_bits;
        float value_a = fa->value;

        {
            float * wl = lin_weights + index_a * field_stride;
            float * wgl = wl + n_dim_aligned;

            __m256 ymm_val = _mm256_set1_ps(value_a / linear_norm);

            for (uint d = 0; d < n_dim; d += 8) {
                __m256 ymm_kappa_val = _mm256_load_ps(l0_output_grad + d) * ymm_val;

                // Load weights
                __m256 ymm_wl = _mm256_load_ps(wl + d);
                __m256 ymm_wgl = _mm256_load_ps(wgl + d);

                // Compute gradient values
                __m256 ymm_g  = ymm_ffm_lambda * ymm_wl + ymm_kappa_val;

                // Update weights
                ymm_wgl = ymm_wgl + ymm_g * ymm_g;
                ymm_wl  = ymm_wl - ymm_eta * ymm_g * _mm256_rsqrt_ps(ymm_wgl);

                // Store weights
                _mm256_store_ps(wl + d, ymm_wl);
                _mm256_store_ps(wgl + d, ymm_wgl);
            }
        }

        if (field_a < min_a_field)
            continue;

        for (const ffm_feature * fb = start; fb != fa; ++ fb) {
            uint index_b = fb->index &  ffm_hash_mask;
            uint field_b = fb->index >> ffm_hash_bits;
            float value_b = fb->value;

            if (field_b > max_b_field)
                break;

            if (fb + prefetch_depth < fa && test_mask_bit(dropout_mask, dropout_idx + prefetch_depth)) { // Prefetch row only if no dropout
                uint index_p = fb[prefetch_depth].index &  ffm_hash_mask;
                uint field_p = fb[prefetch_depth].index >> ffm_hash_bits;

                prefetch_interaction_weights(ffm_weights + index_p * index_stride + field_a * field_stride);
                prefetch_interaction_weights(ffm_weights + index_a * index_stride + field_p * field_stride);
            }

            if (test_mask_bit(dropout_mask, dropout_idx ++) == 0)
                continue;

            float * wa = ffm_weights + index_a * index_stride + field_b * field_stride;
            float * wb = ffm_weights + index_b * index_stride + field_a * field_stride;

            float * wga = wa + n_dim_aligned;
            float * wgb = wb + n_dim_aligned;

            __m256 ymm_val = _mm256_set1_ps(dropout_mult * value_a * value_b / norm);

            for (uint d = 0; d < n_dim; d += 8) {
                __m256 ymm_kappa_val = _mm256_load_ps(l0_output_grad + d) * ymm_val;

                // Load weights
                __m256 ymm_wa = _mm256_load_ps(wa + d);
                __m256 ymm_wb = _mm256_load_ps(wb + d);

                __m256 ymm_wga = _mm256_load_ps(wga + d);
                __m256 ymm_wgb = _mm256_load_ps(wgb + d);

                // Compute gradient values
                __m256 ymm_ga = ymm_ffm_lambda * ymm_wa + ymm_kappa_val * ymm_wb;
                __m256 ymm_gb = ymm_ffm_lambda * ymm_wb + ymm_kappa_val * ymm_wa;

                // Update weights
                ymm_wga = _mm256_add_ps(ymm_wga, ymm_ga * ymm_ga);
                ymm_wgb = _mm256_add_ps(ymm_wgb, ymm_gb * ymm_gb);

                ymm_wa  = _mm256_sub_ps(ymm_wa, ymm_eta * ymm_ga * _mm256_rsqrt_ps(ymm_wga));
                ymm_wb  = _mm256_sub_ps(ymm_wb, ymm_eta * ymm_gb * _mm256_rsqrt_ps(ymm_wgb));

                // Store weights
                _mm256_store_ps(wa + d, ymm_wa);
                _mm256_store_ps(wb + d, ymm_wb);

                _mm256_store_ps(wga + d, ymm_wga);
                _mm256_store_ps(wgb + d, ymm_wgb);
            }
        }
    }
}
