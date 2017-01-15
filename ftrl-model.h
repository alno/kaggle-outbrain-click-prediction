#pragma once

#include "ffm.h"

class ftrl_model {
    float * weights_z;
    float * weights_n;

    float alpha, beta, l1, l2;

    uint n_bits, n_weights, mask;
public:
    ftrl_model(uint n_bits, float alpha, float beta, float l1, float l2);
    ~ftrl_model();

    ffm_float predict(const ffm_feature * start, const ffm_feature * end, ffm_float norm, uint64_t * dropout_mask, float dropout_mult);
    void update(const ffm_feature * start, const ffm_feature * end, ffm_float norm, ffm_float kappa, uint64_t * dropout_mask, float dropout_mult);

    uint get_dropout_mask_size(const ffm_feature * start, const ffm_feature * end) { return 0; }
};
