#include "ftrl-model.h"
#include "util/model-helpers.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <algorithm>


template <typename T>
inline int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}


template <typename T>
class local_weight_vector {
    T * w;
    uint sz;
public:
    local_weight_vector() {
        w = nullptr;
        sz = 0;
    }

    ~local_weight_vector() {
        if (w != nullptr) free(w);
    }

    T * get(uint size) {
        if (w == nullptr) {
            w = malloc_aligned<T>(size);

            zero_weights(w, size);

            sz = size;
        } else if (sz != size) {
            throw std::logic_error("Weights are already initialized with other size!");
        }

        return w;
    }
};


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


static thread_local local_weight_vector<float> local_weights;
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

    zero_weights(weights_z, n_weights);
    zero_weights(weights_n, n_weights);
}

ftrl_model::~ftrl_model() {
    free(weights_z);
    free(weights_n);
}



float ftrl_model::predict(const ffm_feature * start, const ffm_feature * end, ffm_float norm, uint64_t * dropout_mask) {
    auto & feature_buf = local_feature_buffer;

    feature_buf.clear();
    feature_buf.add(0, 1.0);

    int i = 0;
    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        feature_buf.add(fa->index & mask, fa->value);

        if ((fa->index >> ffm_hash_bits) < min_a_field)
            continue;

        for (const ffm_feature * fb = start; fb != fa; ++ fb, ++ i) {
            if ((fb->index >> ffm_hash_bits) > max_b_field)
                break;

            if (test_mask_bit(dropout_mask, i) == 0)
                continue;

            feature_buf.add((fa->index + fb->index * 2654435761) & mask, fa->value * fb->value);
        }
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


void ftrl_model::update(const ffm_feature * start, const ffm_feature * end, ffm_float norm, ffm_float grad, uint64_t * dropout_mask) {
    auto & feature_buf = local_feature_buffer;

    uint feature_count = feature_buf.size;
    uint * feature_indices = feature_buf.indices;
    float * feature_values = feature_buf.values;
    float * feature_weights = feature_buf.weights;

    for (uint i = 0; i < feature_count; ++ i) {
        uint  feature_index = feature_indices[i];
        float feature_grad = grad * feature_values[i];

        float feature_grad_sqr = feature_grad * feature_grad;

        float n = weights_n[feature_index];

        float sigma = (sqrt(n + feature_grad_sqr) - sqrt(n)) / alpha;

        weights_z[feature_index] += feature_grad - sigma * feature_weights[i];
        weights_n[feature_index] += feature_grad_sqr;
    }
}
