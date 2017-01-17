#include "nn-model.h"

#include "util/model-helpers.h"
#include "util/nn-helpers.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>


constexpr uint n_features = 1 << ffm_hash_bits;

constexpr uint l0_output_size = aligned_float_array_size(96);
constexpr uint l1_output_size = aligned_float_array_size(64);
constexpr uint l2_output_size = aligned_float_array_size(48);

constexpr uint l1_layer_size = l0_output_size * (l1_output_size - 1);
constexpr uint l2_layer_size = l1_output_size * (l2_output_size - 1);
constexpr uint l3_layer_size = l2_output_size;


uint rehash(uint feature_index) {
    return feature_index &  ffm_hash_mask;
}


class state_buffer {
public:
    float * l0_output;
    float * l0_output_grad;
    float * l0_dropout_mask;

    float * l1_output;
    float * l1_output_grad;
    float * l1_dropout_mask;

    float * l2_output;
    float * l2_output_grad;
    float * l2_dropout_mask;

    std::default_random_engine gen;
public:
    state_buffer() {
        l0_output = malloc_aligned<float>(l0_output_size);
        l0_output_grad = malloc_aligned<float>(l0_output_size);
        l0_dropout_mask = malloc_aligned<float>(l0_output_size);

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
        free(l0_dropout_mask);

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

    lin_w = malloc_aligned<float>(n_features * l0_output_size);
    lin_wg = malloc_aligned<float>(n_features * l0_output_size);

    l1_w = malloc_aligned<float>(l1_layer_size);
    l1_wg = malloc_aligned<float>(l1_layer_size);

    l2_w = malloc_aligned<float>(l2_layer_size);
    l2_wg = malloc_aligned<float>(l2_layer_size);

    l3_w = malloc_aligned<float>(l3_layer_size);
    l3_wg = malloc_aligned<float>(l3_layer_size);

    fill_with_rand(lin_w, n_features * l0_output_size, std::uniform_real_distribution<float>(-0.1, 0.1), rnd);
    fill_with_ones(lin_wg, n_features * l0_output_size);

    fill_with_rand(l1_w, l1_layer_size, std::normal_distribution<float>(0, 2/sqrt(l0_output_size)), rnd);
    fill_with_ones(l1_wg, l1_layer_size);

    fill_with_rand(l2_w, l2_layer_size, std::normal_distribution<float>(0, 2/sqrt(l1_output_size)), rnd);
    fill_with_ones(l2_wg, l2_layer_size);

    fill_with_rand(l3_w, l3_layer_size, std::normal_distribution<float>(0, 2/sqrt(l2_output_size)), rnd);
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
    return 0;
}


float nn_model::predict(const ffm_feature * start, const ffm_feature * end, float norm, uint64_t * _dropout_mask, float dropout_mult) {
    float linear_norm = end - start;
    state_buffer & buf = local_state_buffer;

    float * l0_output = buf.l0_output;
    float * l0_dropout_mask = buf.l0_dropout_mask;

    float * l1_output = buf.l1_output;
    float * l1_dropout_mask = buf.l1_dropout_mask;

    float * l2_output = buf.l2_output;
    float * l2_dropout_mask = buf.l2_dropout_mask;

    auto & gen = buf.gen;

    std::uniform_real_distribution<float> dropout_distr(0, 1);

    if (dropout_mult > 1) { // Apply dropout only in train
        float l0_dropout_prob = 0;//0.02;
        float l1_dropout_prob = 0;//0.02;
        float l2_dropout_prob = 0;//0.02;

        float l0_dropout_scale = 1 / (1 - l0_dropout_prob);
        float l1_dropout_scale = 1 / (1 - l1_dropout_prob);
        float l2_dropout_scale = 1 / (1 - l2_dropout_prob);

        // Prepare dropout masks
        l0_dropout_mask[0] = 1.0; // No dropout on bias
        for (uint j = 1; j < l0_output_size; ++ j)
            l0_dropout_mask[j] = (dropout_distr(gen) >= l0_dropout_prob) * l0_dropout_scale;

        l1_dropout_mask[0] = 1.0; // No dropout on bias
        for (uint j = 1; j < l1_output_size; ++ j)
            l1_dropout_mask[j] = (dropout_distr(gen) >= l1_dropout_prob) * l1_dropout_scale;

        l2_dropout_mask[0] = 1.0; // No dropout on bias
        for (uint j = 1; j < l2_output_size; ++ j)
            l2_dropout_mask[j] = (dropout_distr(gen) >= l2_dropout_prob) * l2_dropout_scale;
    } else {
        fill_with_ones(l0_dropout_mask, l0_output_size);
        fill_with_ones(l1_dropout_mask, l1_output_size);
        fill_with_ones(l2_dropout_mask, l2_output_size);
    }

    // Compute activations

    fill_with_zero(l0_output, l0_output_size);
    fill_with_zero(l1_output, l1_output_size);
    fill_with_zero(l2_output, l2_output_size);

    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        uint index = rehash(fa->index);
        float value = fa->value;

        float * wl = lin_w + index * l0_output_size;

        __m256 ymm_val = _mm256_set1_ps(value / linear_norm);
        for(ffm_uint d = 0; d < l0_output_size; d += 8) {
            _mm256_store_ps(l0_output + d,  _mm256_load_ps(l0_output + d) + _mm256_load_ps(wl + d) * ymm_val);
        }
    }

    l0_output[0] = 1.0; // Layer 0 bias, here we rewritre some computation results, but who cares
    l1_output[0] = 1.0; // Layer 1 bias
    l2_output[0] = 1.0; // Layer 2 bias

    // Layer 0 relu
    for (uint j = 1; j < l0_output_size; ++ j)
        l0_output[j] = relu(l0_output[j]) * l0_dropout_mask[j];

    // Layer 1 forward pass
    for (uint j = 1; j < l1_output_size; ++ j)
        l1_output[j] = relu(forward_pass(l0_output_size, l0_output, l1_w + (j - 1) * l0_output_size)) * l1_dropout_mask[j];

    // Layer 2 forward pass
    for (uint j = 1; j < l2_output_size; ++ j)
        l2_output[j] = relu(forward_pass(l1_output_size, l1_output, l2_w + (j - 1) * l1_output_size)) * l2_dropout_mask[j];

    // Layer 3 forward pass
    return forward_pass(l2_output_size, l2_output, l3_w);
}


void nn_model::update(const ffm_feature * start, const ffm_feature * end, float norm, float kappa, uint64_t * _dropout_mask, float _dropout_mult) {
    float linear_norm = end - start;
    state_buffer & buf = local_state_buffer;

    float * l0_output = buf.l0_output;
    float * l0_output_grad = buf.l0_output_grad;
    float * l0_dropout_mask = buf.l0_dropout_mask;

    float * l1_output = buf.l1_output;
    float * l1_output_grad = buf.l1_output_grad;
    float * l1_dropout_mask = buf.l1_dropout_mask;

    float * l2_output = buf.l2_output;
    float * l2_output_grad = buf.l2_output_grad;
    float * l2_dropout_mask = buf.l2_dropout_mask;

    fill_with_zero(l0_output_grad, l0_output_size);
    fill_with_zero(l1_output_grad, l1_output_size);
    fill_with_zero(l2_output_grad, l2_output_size);

    backward_pass(l2_output_size, l2_output, l2_output_grad, l3_w, l3_wg, kappa, eta, lambda);

    // Backprop layer 2
    for (uint j = 1, ofs = 0; j < l2_output_size; ++ j, ofs += l1_output_size) {
        float l2_grad = l2_output_grad[j] * l2_dropout_mask[j];

        if (l2_output[j] <= 0) // Relu activation: grad in negative part is zero
            l2_grad = 0;

        backward_pass(l1_output_size, l1_output, l1_output_grad, l2_w + ofs, l2_wg + ofs, l2_grad, eta, lambda);
    }

    // Backprop layer 1
    for (uint j = 1, ofs = 0; j < l1_output_size; ++ j, ofs += l0_output_size) {
        float l1_grad = l1_output_grad[j] * l1_dropout_mask[j];

        if (l1_output[j] <= 0) // Relu activation: grad in negative part is zero
            l1_grad = 0;

        backward_pass(l0_output_size, l0_output, l0_output_grad, l1_w + ofs, l1_wg + ofs, l1_grad, eta, lambda);
    }

    // Backprop layer 0
    l0_output_grad[0] = 0;
    for (uint j = 1; j < l0_output_size; ++ j) {
        float l0_grad = l0_output_grad[j] * l0_dropout_mask[j];

        if (l0_output[j] <= 0) // Relu activation: grad in negative part is zero
            l0_grad = 0;

        l0_output_grad[j] = l0_grad;
    }

    // Update linear and interaction weights
    __m256 ymm_eta = _mm256_set1_ps(eta);
    __m256 ymm_lambda = _mm256_set1_ps(lambda);

    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        uint index = rehash(fa->index);
        float value = fa->value;

        float * wl = lin_w + index * l0_output_size;
        float * wgl = lin_wg + index * l0_output_size;

        __m256 ymm_val = _mm256_set1_ps(value / linear_norm);

        for (uint d = 0; d < l0_output_size; d += 8) {
            __m256 ymm_kappa_val = _mm256_load_ps(l0_output_grad + d) * ymm_val;

            // Load weights
            __m256 ymm_wl = _mm256_load_ps(wl + d);
            __m256 ymm_wgl = _mm256_load_ps(wgl + d);

            // Compute gradient values
            __m256 ymm_g  = ymm_lambda * ymm_wl + ymm_kappa_val;

            // Update weights
            ymm_wgl = ymm_wgl + ymm_g * ymm_g;
            ymm_wl  = ymm_wl - ymm_eta * ymm_g * _mm256_rsqrt_ps(ymm_wgl);

            // Store weights
            _mm256_store_ps(wl + d, ymm_wl);
            _mm256_store_ps(wgl + d, ymm_wgl);
        }
    }
}
