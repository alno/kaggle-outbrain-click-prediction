#pragma once

#include "ffm.h"

class ffm_nn_model {
    float * ffm_weights;
    float * lin_weights;

    float * l1_w;
    float * l1_wg;

    float * l2_w;
    float * l2_wg;

    float eta, ffm_lambda, nn_lambda;

    uint max_b_field, min_a_field;
public:
    ffm_nn_model(int seed, bool restricted, float eta, float ffm_lambda, float nn_lambda);
    ~ffm_nn_model();

    ffm_float predict(const ffm_feature * start, const ffm_feature * end, float norm, uint64_t * dropout_mask, float dropout_mult);
    void update(const ffm_feature * start, const ffm_feature * end, float norm, float kappa, uint64_t * dropout_mask, float dropout_mult);

    uint get_dropout_mask_size(const ffm_feature * start, const ffm_feature * end);
};
