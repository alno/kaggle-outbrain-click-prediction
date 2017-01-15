#pragma once

#include <random>

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

constexpr uint aligned_float_array_size(uint cnt) {
    return ((cnt - 1) / align_floats + 1) * align_floats;
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
inline void fill_with_zero(T * weights, size_t n) {
    T * w = weights;

    for(size_t i = 0; i < n; i++)
        *w++ = T(0);
}


template <typename D>
static void fill_with_rand(ffm_float * weights, ffm_uint n, D gen, std::default_random_engine & rnd) {
    ffm_float * w = weights;

    for(ffm_uint i = 0; i < n; i++) {
        *w++ = gen(rnd);
    }
}


template <typename T>
inline void fill_with_ones(T * weights, size_t n) {
    T * w = weights;

    for(size_t i = 0; i < n; i++)
        *w++ = T(1);
}


inline uint test_mask_bit(uint64_t * mask, uint i) {
    return (mask[i >> 6] >> (i & 63)) & 1;
}


template <typename T>
inline T min(T a, T b) {
    return a < b ? a : b;
}


template <typename T>
inline int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}


inline float relu(float val) {
    return val > 0 ? val : 0;
}

inline bool isnan(float val) {
    return val != val;
}
