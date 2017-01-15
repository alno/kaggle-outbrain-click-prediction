#pragma once

#include <immintrin.h>

// Define intrinsic missing in gcc
#define _mm256_set_m128(v0, v1)  _mm256_insertf128_ps(_mm256_castps128_ps256(v1), (v0), 1)



constexpr ffm_uint align_bytes = 32;
constexpr ffm_uint align_floats = align_bytes / sizeof(float);


inline float sum(__m256 val) {
    __m128 s = _mm256_extractf128_ps(_mm256_add_ps(val,  _mm256_permute2f128_ps(val, val, 1)), 0);

    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);

    float sum;
    _mm_store_ss(&sum, s);

    return sum;
}


template <typename T>
inline T * malloc_aligned(size_t size) {
    void *ptr;

    int status = posix_memalign(&ptr, align_bytes, size*sizeof(T));

    if(status != 0)
        throw std::bad_alloc();

    return (T*) ptr;
}


template <typename T>
inline void zero_weights(T * weights, size_t n) {
    T * w = weights;

    for(size_t i = 0; i < n; i++)
        *w++ = T(0);
}


inline uint test_mask_bit(uint64_t * mask, uint i) {
    return (mask[i >> 6] >> (i & 63)) & 1;
}
