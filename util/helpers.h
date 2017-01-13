#pragma once

#include <cmath>

constexpr float pos_time_diff(int64_t td) {
    if (td < 0)
        return 0;

    return log(1 + td) / 100;
}

constexpr float time_diff(int64_t td) {
    if (td < 0)
        return - log(1 - td) / 100;

    return log(1 + td) / 100;
}

constexpr float logit(float p) {
    return log(p / (1-p));
}

constexpr float base_ctr = 0.194f;
constexpr float base_ctr_logit = logit(base_ctr);

constexpr float ctr_logit(float views, float clicks, float reg = 50) {
    return logit((clicks + base_ctr * reg) / (views + reg)) - base_ctr_logit;
}


template <typename T>
inline T min(T a, T b) {
    return a < b ? a : b;
}
