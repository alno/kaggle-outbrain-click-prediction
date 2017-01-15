#include "ffm-nn-model.h"
#include "util/model-helpers.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>


constexpr ffm_ulong n_fields = 40;
constexpr ffm_ulong n_features = 1 << ffm_hash_bits;

constexpr ffm_ulong n_dim = 14;
constexpr ffm_ulong n_dim_aligned = ((n_dim - 1) / align_floats + 1) * align_floats;

constexpr ffm_ulong index_stride = n_fields * n_dim_aligned * 2;
constexpr ffm_ulong field_stride = n_dim_aligned * 2;

constexpr uint prefetch_depth = 1;


constexpr uint interaction_output_size = 50;

constexpr uint l0_output_size = aligned_float_array_size(1 + n_fields + interaction_output_size);
constexpr uint l1_output_size = aligned_float_array_size(24);

constexpr uint l1_layer_size = l0_output_size * (l1_output_size - 1);
constexpr uint l2_layer_size = l1_output_size;


class state_buffer {
public:
    float * l0_output;
    float * l0_output_grad;

    float * l1_output;
    float * l1_output_grad;
public:
    state_buffer() {
        l0_output = malloc_aligned<float>(l0_output_size);
        l0_output_grad = malloc_aligned<float>(l0_output_size);

        l1_output = malloc_aligned<float>(l1_output_size);
        l1_output_grad = malloc_aligned<float>(l1_output_size);
    }

    ~state_buffer() {
        free(l0_output);
        free(l0_output_grad);

        free(l1_output);
        free(l1_output_grad);
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

    lin_w = malloc_aligned<float>(n_features);
    lin_wg = malloc_aligned<float>(n_features);

    l1_w = malloc_aligned<float>(l1_layer_size);
    l1_wg = malloc_aligned<float>(l1_layer_size);

    l2_w = malloc_aligned<float>(l2_layer_size);
    l2_wg = malloc_aligned<float>(l2_layer_size);

    init_interaction_weights(ffm_weights, n_features * n_fields, std::uniform_real_distribution<float>(-1.0/sqrt(n_dim), 1.0/sqrt(n_dim)), rnd);

    fill_with_rand(lin_w, n_features, std::uniform_real_distribution<float>(-1e-3, 1e-3), rnd);
    fill_with_ones(lin_wg, n_features);

    fill_with_rand(l1_w, l1_layer_size, std::uniform_real_distribution<float>(-1.0/l1_output_size, 1.0/l1_output_size), rnd);
    fill_with_ones(l1_wg, l1_layer_size);

    fill_with_rand(l2_w, l2_layer_size, std::uniform_real_distribution<float>(-1.0, 1.0), rnd);
    fill_with_ones(l2_wg, l2_layer_size);
}


ffm_nn_model::~ffm_nn_model() {
    free(ffm_weights);

    free(lin_w);
    free(lin_wg);

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
    float * l1_output = local_state_buffer.l1_output;

    fill_with_zero(l0_output, l0_output_size);
    fill_with_zero(l1_output, l1_output_size);

    l0_output[0] = 1.0; // Layer 1 bias
    l1_output[0] = 1.0; // Layer 2 bias

    uint dropout_idx = 0;
    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        uint index_a = fa->index &  ffm_hash_mask;
        uint field_a = fa->index >> ffm_hash_bits;
        float value_a = fa->value;

        l0_output[1 + field_a] += lin_w[index_a] * value_a / linear_norm;

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

            //if (field_a == field_b)
            //    continue;

            ffm_float * wa = ffm_weights + index_a * index_stride + field_b * field_stride;
            ffm_float * wb = ffm_weights + index_b * index_stride + field_a * field_stride;

            __m256 ymm_total = _mm256_set1_ps(0);
            __m256 ymm_val = _mm256_set1_ps(dropout_mult * value_a * value_b / norm);

            for(ffm_uint d = 0; d < n_dim; d += 8) {
                __m256 ymm_wa = _mm256_load_ps(wa + d);
                __m256 ymm_wb = _mm256_load_ps(wb + d);

                ymm_total = _mm256_add_ps(ymm_total, _mm256_mul_ps(_mm256_mul_ps(ymm_wa, ymm_wb), ymm_val));
            }

            l0_output[1 + n_fields + (field_a * 2654435761 + field_b) % interaction_output_size] += sum(ymm_total);
        }
    }

    // Layer 1 forward prop
    float total = l2_w[0];

    for (uint j = 1; j < l1_output_size; ++ j) {
        __m256 ymm_total = _mm256_set1_ps(0);

        for (uint i = 0; i < l0_output_size; i += 8) {
            ymm_total += _mm256_load_ps(l0_output + i) * _mm256_load_ps(l1_w + (j - 1) * l0_output_size + i);
        }

        float l1_out = relu(sum(ymm_total));

        l1_output[j] = l1_out;
        total += l1_out * l2_w[j];
    }

    return total;
}


void ffm_nn_model::update(const ffm_feature * start, const ffm_feature * end, float norm, float kappa, uint64_t * dropout_mask, float dropout_mult) {
    float linear_norm = end - start;

    float * l0_output = local_state_buffer.l0_output;
    float * l0_output_grad = local_state_buffer.l0_output_grad;

    float * l1_output = local_state_buffer.l1_output;
    float * l1_output_grad = local_state_buffer.l1_output_grad;

    fill_with_zero(l0_output_grad, l0_output_size);
    fill_with_zero(l1_output_grad, l1_output_size);

    __m256 ymm_eta = _mm256_set1_ps(eta);
    __m256 ymm_ffm_lambda = _mm256_set1_ps(ffm_lambda);
    __m256 ymm_nn_lambda = _mm256_set1_ps(nn_lambda);
    __m256 ymm_grad = _mm256_set1_ps(kappa);

    // Backprop layer 2
    for (uint i = 0; i < l1_output_size; i += 8) {
        __m256 ymm_w = _mm256_load_ps(l2_w + i);

        __m256 ymm_g = ymm_nn_lambda * ymm_w + ymm_grad * _mm256_load_ps(l1_output + i);
        __m256 ymm_wg = _mm256_load_ps(l2_wg + i) + ymm_g * ymm_g;

        _mm256_store_ps(l1_output_grad + i, ymm_grad * ymm_w);

        _mm256_store_ps(l2_w + i, ymm_w - ymm_eta * ymm_g * _mm256_rsqrt_ps(ymm_wg));
        _mm256_store_ps(l2_wg + i, ymm_wg);
    }

    // Backprop layer 1
    for (uint j = 1; j < l1_output_size; ++ j) {
        __m256 ymm_l1_grad = _mm256_set1_ps(l1_output_grad[j]);

        for (uint i = 0; i < l0_output_size; i += 8) {
            uint ofs = (j - 1) * l0_output_size + i;

            __m256 ymm_w = _mm256_load_ps(l1_w + ofs);

            __m256 ymm_g = ymm_nn_lambda * ymm_w + ymm_l1_grad * _mm256_load_ps(l0_output + i);
            __m256 ymm_wg = _mm256_load_ps(l1_wg + ofs) + ymm_g * ymm_g;

            _mm256_store_ps(l0_output_grad + i, ymm_l1_grad * ymm_w + _mm256_load_ps(l0_output_grad + i));

            _mm256_store_ps(l1_w + ofs, ymm_w - ymm_eta * ymm_g * _mm256_rsqrt_ps(ymm_wg));
            _mm256_store_ps(l1_wg + ofs, ymm_wg);
        }
    }

    // Update linear and interaction weights
    uint dropout_idx = 0;
    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        uint index_a = fa->index &  ffm_hash_mask;
        uint field_a = fa->index >> ffm_hash_bits;
        float value_a = fa->value;

        float g = ffm_lambda * lin_w[index_a] + l0_output_grad[1 + field_a] * value_a / linear_norm;
        float wg = lin_wg[index_a] + g*g;

        lin_w[index_a] -= eta * g / sqrt(wg);
        lin_wg[index_a] = wg;

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

            //if (field_a == field_b)
            //    continue;

            float * wa = ffm_weights + index_a * index_stride + field_b * field_stride;
            float * wb = ffm_weights + index_b * index_stride + field_a * field_stride;

            float * wga = wa + n_dim_aligned;
            float * wgb = wb + n_dim_aligned;

            __m256 ymm_kappa_val = _mm256_set1_ps(l0_output_grad[1 + n_fields + (field_a * 2654435761 + field_b) % interaction_output_size] * dropout_mult * value_a * value_b / norm);

            for(ffm_uint d = 0; d < n_dim; d += 8) {
                // Load weights
                __m256 ymm_wa = _mm256_load_ps(wa + d);
                __m256 ymm_wb = _mm256_load_ps(wb + d);

                __m256 ymm_wga = _mm256_load_ps(wga + d);
                __m256 ymm_wgb = _mm256_load_ps(wgb + d);

                // Compute gradient values
                __m256 ymm_ga = _mm256_add_ps(ymm_ffm_lambda * ymm_wa, ymm_kappa_val * ymm_wb);
                __m256 ymm_gb = _mm256_add_ps(ymm_ffm_lambda * ymm_wb, ymm_kappa_val * ymm_wa);

                // Update weights
                ymm_wga = _mm256_add_ps(ymm_wga, _mm256_mul_ps(ymm_ga, ymm_ga));
                ymm_wgb = _mm256_add_ps(ymm_wgb, _mm256_mul_ps(ymm_gb, ymm_gb));

                ymm_wa  = _mm256_sub_ps(ymm_wa, _mm256_mul_ps(ymm_eta, _mm256_mul_ps(_mm256_rsqrt_ps(ymm_wga), ymm_ga)));
                ymm_wb  = _mm256_sub_ps(ymm_wb, _mm256_mul_ps(ymm_eta, _mm256_mul_ps(_mm256_rsqrt_ps(ymm_wgb), ymm_gb)));

                // Store weights
                _mm256_store_ps(wa + d, ymm_wa);
                _mm256_store_ps(wb + d, ymm_wb);

                _mm256_store_ps(wga + d, ymm_wga);
                _mm256_store_ps(wgb + d, ymm_wgb);
            }
        }
    }
}
