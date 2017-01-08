#pragma once

#include "ffm.h"

class ffm_model {
    ffm_float * weights;
    ffm_float * linear_weights;

    ffm_float bias_w;
    ffm_float bias_wg;

    ffm_float eta;
    ffm_float lambda;

    ffm_uint max_b_field;
    ffm_uint min_a_field;
public:
    ffm_model(int seed, bool restricted, float eta = 0.2, float lambda = 0.00002);
    ~ffm_model();

    ffm_float predict(const ffm_feature * start, const ffm_feature * end, ffm_float norm, uint64_t * mask);
    void update(const ffm_feature * start, const ffm_feature * end, ffm_float norm, ffm_float kappa, uint64_t * mask);
};
