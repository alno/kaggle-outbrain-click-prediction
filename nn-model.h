#pragma once

#include "ffm.h"

class nn_model {
    float * lin_w;
    float * lin_wg;

    float * l1_w;
    float * l1_wg;

    float * l2_w;
    float * l2_wg;

    float * l3_w;
    float * l3_wg;

    float eta;
    float lambda;

    uint max_b_field;
    uint min_a_field;
public:
    nn_model(int seed, float eta, float lambda);
    ~nn_model();

    float predict(const ffm_feature * start, const ffm_feature * end, float norm, uint64_t * dropout_mask, float dropout_mult);
    void update(const ffm_feature * start, const ffm_feature * end, float norm, float kappa, uint64_t * dropout_mask, float dropout_mult);

    uint get_dropout_mask_size(const ffm_feature * start, const ffm_feature * end);
};
