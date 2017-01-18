#pragma once

#include "ffm.h"

class ffm_model {
    float * ffm_weights;
    float * lin_weights;

    float bias_w;
    float bias_wg;

    float eta;
    float lambda;

    ffm_uint max_b_field;
    ffm_uint min_a_field;
public:
    ffm_model(int seed, bool restricted, float eta, float lambda);
    ~ffm_model();

    float predict(const ffm_feature * start, const ffm_feature * end, float norm, uint64_t * dropout_mask, float dropout_mult);
    void update(const ffm_feature * start, const ffm_feature * end, float norm, float kappa, uint64_t * dropout_mask, float dropout_mult);

    uint get_dropout_mask_size(const ffm_feature * start, const ffm_feature * end);
};
