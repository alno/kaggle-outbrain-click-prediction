#include "ftrl-model.h"
#include "util/model-helpers.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>


constexpr uint feature_buffer_size = 100000;


uint max_b_field = 29;
uint min_a_field = 10;


class feature_buffer {
public:
    uint size;

    uint * indices;
    float * values;
    float * weights;
public:
    feature_buffer() {
        size = 0;
        indices = malloc_aligned<uint>(feature_buffer_size);
        values = malloc_aligned<float>(feature_buffer_size);
        weights = malloc_aligned<float>(feature_buffer_size);
    }

    ~feature_buffer() {
        free(indices);
        free(values);
        free(weights);
    }

    void clear() {
        size = 0;
    }

    void add(uint index, float value) {
        indices[size] = index;
        values[size] = value;
        size ++;
    }
};


static thread_local feature_buffer local_feature_buffer;


ftrl_model::ftrl_model(uint n_bits, float alpha, float beta, float l1, float l2) {
    this->alpha = alpha;
    this->beta = beta;
    this->l1 = l1;
    this->l2 = l2;
    this->n_bits = n_bits;

    n_weights = 1 << n_bits;
    mask = n_weights - 1;

    weights_z = malloc_aligned<float>(n_weights);
    weights_n = malloc_aligned<float>(n_weights);

    fill_with_zero(weights_z, n_weights);
    fill_with_zero(weights_n, n_weights);
}

ftrl_model::~ftrl_model() {
    free(weights_z);
    free(weights_n);
}



float ftrl_model::predict(const ffm_feature * start, const ffm_feature * end, ffm_float norm, uint64_t * dropout_mask, float dropout_mult) {
    auto & feature_buf = local_feature_buffer;

    feature_buf.clear();
    feature_buf.add(0, 1.0);

    //int i = 0;
    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        feature_buf.add(fa->index & mask, fa->value);
/*
        if ((fa->index >> ffm_hash_bits) < min_a_field)
            continue;

        for (const ffm_feature * fb = start; fb != fa; ++ fb, ++ i) {
            if ((fb->index >> ffm_hash_bits) > max_b_field)
                break;

            if (test_mask_bit(dropout_mask, i) == 0)
                continue;

            feature_buf.add((fa->index + fb->index * 2654435761) & mask, fa->value * fb->value);
        }*/
    }

    uint feature_count = feature_buf.size;
    uint * feature_indices = feature_buf.indices;
    float * feature_values = feature_buf.values;
    float * feature_weights = feature_buf.weights;

    float total = 0;

    for (uint i = 0; i < feature_count; ++ i) {
        uint feature_index = feature_indices[i];

        float zi = weights_z[feature_index];
        float zsi = sgn(zi);

        if (zsi * zi < l1) {
            feature_weights[i] = 0;
        } else {
            float wi = (zsi * l1 - zi) * feature_values[i] / ((beta + sqrt(weights_n[feature_index])) / alpha + l2);

            feature_weights[i] = wi;

            total += wi;
        }
    }

    return total;
}


void ftrl_model::update(const ffm_feature * start, const ffm_feature * end, ffm_float norm, ffm_float grad, uint64_t * dropout_mask, float dropout_mult) {
    auto & feature_buf = local_feature_buffer;

    uint feature_count = feature_buf.size;

    uint  * fi = feature_buf.indices;
    float * fv = feature_buf.values;
    float * fw = feature_buf.weights;

    float * n = weights_n;

    __m256 ymm_alpha = _mm256_set1_ps(alpha);
    __m256 ymm_grad = _mm256_set1_ps(grad);

    for (uint i = 0; i < feature_count; i += 8) {
        __m256 ymm_n = _mm256_set_ps(n[fi[i + 7]], n[fi[i + 6]], n[fi[i + 5]], n[fi[i + 4]], n[fi[i + 3]], n[fi[i + 2]], n[fi[i + 1]], n[fi[i]]);

        __m256 ymm_fg = _mm256_load_ps(fv + i) * ymm_grad;
        __m256 ymm_fg_sqr = ymm_fg * ymm_fg;

        __m256 ymm_sigma = (_mm256_sqrt_ps(ymm_n + ymm_fg_sqr) - _mm256_sqrt_ps(ymm_n)) / ymm_alpha;

        __m256 ymm_za = ymm_fg - ymm_sigma * _mm256_load_ps(fw + i);

        float * za = (float *)(&ymm_za);
        float * gs = (float *)(&ymm_fg_sqr);

        uint fl = min(8u, feature_count - i);

        for (uint j = 0; j < fl; ++ j) {
            weights_z[fi[i + j]] += za[j];
            weights_n[fi[i + j]] += gs[j];
        }
    }
}
