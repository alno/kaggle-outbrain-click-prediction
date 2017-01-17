#pragma once

#include "model-helpers.h"


inline void backward_pass(uint input_size, float * input, float * input_grad, float * w, float * wg, float grad, float eta, float lambda) {
    __m256 ymm_eta = _mm256_set1_ps(eta);
    __m256 ymm_lambda = _mm256_set1_ps(lambda);
    __m256 ymm_grad = _mm256_set1_ps(grad);

    for (uint i = 0; i < input_size; i += 8) {
        __m256 ymm_w = _mm256_load_ps(w + i);

        __m256 ymm_g = ymm_lambda * ymm_w + ymm_grad * _mm256_load_ps(input + i);
        __m256 ymm_wg = _mm256_load_ps(wg + i) + ymm_g * ymm_g;

        _mm256_store_ps(input_grad + i, ymm_grad * ymm_w + _mm256_load_ps(input_grad + i));

        _mm256_store_ps(w + i, ymm_w - ymm_eta * ymm_g * _mm256_rsqrt_ps(ymm_wg));
        _mm256_store_ps(wg + i, ymm_wg);
    }
}

inline float forward_pass(uint input_size, float * input, float * w) {
    __m256 ymm_total = _mm256_set1_ps(0);

    for (uint i = 0; i < input_size; i += 8)
        ymm_total += _mm256_load_ps(input + i) * _mm256_load_ps(w + i);

    return sum(ymm_total);
}
