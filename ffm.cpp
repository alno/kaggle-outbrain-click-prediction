#include "ffm.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <algorithm>

#include <pmmintrin.h>
#include <immintrin.h>

#include <omp.h>

#include <boost/program_options.hpp>


const ffm_uint align_bytes = 16;
const ffm_uint align_floats = align_bytes / sizeof(ffm_float);


ffm_float * malloc_aligned_float(ffm_ulong size) {
    void *ptr;

    int status = posix_memalign(&ptr, align_bytes, size*sizeof(ffm_float));

    if(status != 0)
        throw std::bad_alloc();

    return (ffm_float*) ptr;
}

const ffm_uint batch_size = 10000;
const ffm_uint mini_batch_size = 32;

const ffm_ulong n_fields = 30;
const ffm_ulong n_features = 1 << ffm_hash_bits;

const ffm_ulong n_dim = 4;
const ffm_ulong n_dim_aligned = ((n_dim - 1) / align_floats + 1) * align_floats;

const ffm_ulong index_stride = n_fields * n_dim_aligned * 2;
const ffm_ulong field_stride = n_dim_aligned * 2;


// Dropout configuration
const ffm_uint dropout_mask_size = 100; // in 64-bit words
const ffm_uint dropout_prob_log = 1; // 0.5 dropout rate
const ffm_float dropout_norm_mult = ((1 << dropout_prob_log) - 1.0f) / (1 << dropout_prob_log);


std::default_random_engine rnd(2017);


ffm_float * weights;

ffm_float bias_w = 0;
ffm_float bias_wg = 1;


ffm_float eta = 0.2;
ffm_float lambda = 0.00002;

ffm_uint max_b_field = n_fields;
ffm_uint min_a_field = 0;


template <typename T>
T min(T a, T b) {
    return a < b ? a : b;
}

struct ffm_dataset {
    ffm_index index;
    std::string data_file_name;
};


ffm_float predict(const ffm_feature * start, const ffm_feature * end, ffm_float norm, uint64_t * mask) {
    __m128 xmm_total = _mm_set1_ps(bias_w);

    ffm_uint i = 0;

    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        ffm_uint index_a = fa->index &  ffm_hash_mask;
        ffm_uint field_a = fa->index >> ffm_hash_bits;
        ffm_float value_a = fa->value;

        if (field_a < min_a_field)
            continue;

        for (const ffm_feature * fb = start; fb != fa; ++ fb, ++ i) {
            ffm_uint index_b = fb->index &  ffm_hash_mask;
            ffm_uint field_b = fb->index >> ffm_hash_bits;
            ffm_float value_b = fb->value;

            if (field_b > max_b_field)
                break;

            if (((mask[i >> 6] >> (i & 63)) & 1) == 0)
                continue;

            //if (field_a == field_b)
            //    continue;

            ffm_float * wa = weights + index_a * index_stride + field_b * field_stride;
            ffm_float * wb = weights + index_b * index_stride + field_a * field_stride;

            __m128 xmm_val = _mm_set1_ps(value_a * value_b / norm);

            for(ffm_uint d = 0; d < n_dim; d += 4) {
                __m128 xmm_wa = _mm_load_ps(wa + d);
                __m128 xmm_wb = _mm_load_ps(wb + d);

                xmm_total = _mm_add_ps(xmm_total, _mm_mul_ps(_mm_mul_ps(xmm_wa, xmm_wb), xmm_val));
            }
        }
    }

    xmm_total = _mm_hadd_ps(xmm_total, xmm_total);
    xmm_total = _mm_hadd_ps(xmm_total, xmm_total);

    ffm_float total;

    _mm_store_ss(&total, xmm_total);

    return total;
}


void update(const ffm_feature * start, const ffm_feature * end, ffm_float norm, ffm_float kappa, uint64_t * mask) {
    __m128 xmm_eta = _mm_set1_ps(eta);
    __m128 xmm_lambda = _mm_set1_ps(lambda);

    ffm_uint i = 0;

    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        ffm_uint index_a = fa->index &  ffm_hash_mask;
        ffm_uint field_a = fa->index >> ffm_hash_bits;
        ffm_float value_a = fa->value;

        if (field_a < min_a_field)
            continue;

        for (const ffm_feature * fb = start; fb != fa; ++ fb, ++ i) {
            ffm_uint index_b = fb->index &  ffm_hash_mask;
            ffm_uint field_b = fb->index >> ffm_hash_bits;
            ffm_float value_b = fb->value;

            if (field_b > max_b_field)
                break;

            if (((mask[i >> 6] >> (i & 63)) & 1) == 0)
                continue;

            //if (field_a == field_b)
            //    continue;

            ffm_float * wa = weights + index_a * index_stride + field_b * field_stride;
            ffm_float * wb = weights + index_b * index_stride + field_a * field_stride;

            ffm_float * wga = wa + n_dim_aligned;
            ffm_float * wgb = wb + n_dim_aligned;

            __m128 xmm_kappa_val = _mm_set1_ps(kappa * value_a * value_b / norm);

            for(ffm_uint d = 0; d < n_dim; d += 4) {
                // Load weights
                __m128 xmm_wa = _mm_load_ps(wa + d);
                __m128 xmm_wb = _mm_load_ps(wb + d);

                __m128 xmm_wga = _mm_load_ps(wga + d);
                __m128 xmm_wgb = _mm_load_ps(wgb + d);

                // Compute gradient values
                __m128 xmm_ga = _mm_add_ps(_mm_mul_ps(xmm_lambda, xmm_wa), _mm_mul_ps(xmm_kappa_val, xmm_wb));
                __m128 xmm_gb = _mm_add_ps(_mm_mul_ps(xmm_lambda, xmm_wb), _mm_mul_ps(xmm_kappa_val, xmm_wa));

                // Update weights
                xmm_wga = _mm_add_ps(xmm_wga, _mm_mul_ps(xmm_ga, xmm_ga));
                xmm_wgb = _mm_add_ps(xmm_wgb, _mm_mul_ps(xmm_gb, xmm_gb));

                xmm_wa = _mm_sub_ps(xmm_wa, _mm_mul_ps(xmm_eta, _mm_mul_ps(_mm_rsqrt_ps(xmm_wga), xmm_ga)));
                xmm_wb = _mm_sub_ps(xmm_wb, _mm_mul_ps(xmm_eta, _mm_mul_ps(_mm_rsqrt_ps(xmm_wgb), xmm_gb)));

                // Store weights
                _mm_store_ps(wa + d, xmm_wa);
                _mm_store_ps(wb + d, xmm_wb);

                _mm_store_ps(wga + d, xmm_wga);
                _mm_store_ps(wgb + d, xmm_wgb);
            }
        }
    }

    // Update bias
    bias_wg += kappa;
    bias_w -= eta * kappa / sqrt(bias_wg);
}


std::vector<std::pair<ffm_ulong, ffm_ulong>> generate_batches(const ffm_index & index, bool shuffle) {
    std::vector<std::pair<ffm_ulong, ffm_ulong>> batches;

    for (ffm_ulong batch_start = 0; batch_start < index.size; batch_start += batch_size)
        batches.push_back(std::make_pair(batch_start, min(batch_start + batch_size, index.size)));

    if (shuffle)
        std::shuffle(batches.begin(), batches.end(), rnd);

    return batches;
}


std::vector<std::pair<ffm_ulong, ffm_ulong>> generate_mini_batches(ffm_ulong begin, ffm_ulong end) {
    std::vector<std::pair<ffm_ulong, ffm_ulong>> batches;

    for (ffm_ulong mini_batch_start = begin; mini_batch_start < end; mini_batch_start += mini_batch_size)
        batches.push_back(std::make_pair(mini_batch_start, min(mini_batch_start + mini_batch_size, end)));

    std::shuffle(batches.begin(), batches.end(), rnd);

    return batches;
}


void fill_mask_rand(uint64_t * mask, int size, int zero_prob_log) {
    memset(mask, 0, size * sizeof(uint64_t));

    for (int p = 0; p < size; ++ p) {
        for (int i = 0; i < zero_prob_log; ++ i) {
            long long unsigned int v;
            _rdrand64_step(&v);
            mask[p] |= v;
        }
    }
}

void fill_mask_ones(uint64_t * mask, int size) {
    memset(mask, 0xFF, size * sizeof(uint64_t));
}


ffm_double train_on_dataset(const ffm_dataset & dataset) {
    clock_t begin = clock();

    std::cout << "  Training... ";
    std::cout.flush();

    auto batches = generate_batches(dataset.index, true);

    ffm_double loss = 0.0;
    ffm_ulong cnt = 0;

    // Iterate over batches, read each and then iterate over examples
    #pragma omp parallel for schedule(dynamic, 1) reduction(+: loss) reduction(+: cnt)
    for (ffm_ulong bi = 0; bi < batches.size(); ++ bi) {
        auto batch_start_index = batches[bi].first;
        auto batch_end_index = batches[bi].second;

        auto batch_start_offset = dataset.index.offsets[batch_start_index];
        auto batch_end_offset = dataset.index.offsets[batch_end_index];

        auto mini_batches = generate_mini_batches(batch_start_index, batch_end_index);

        std::vector<ffm_feature> batch_features = ffm_read_batch(dataset.data_file_name, batch_start_offset, batch_end_offset);
        ffm_feature * batch_features_data = batch_features.data();

        uint64_t mask[dropout_mask_size];

        for (auto mb = mini_batches.begin(); mb != mini_batches.end(); ++ mb) {
            for (auto ei = mb->first; ei < mb->second; ++ ei) {
                ffm_float y = dataset.index.labels[ei];
                ffm_float norm = dataset.index.norms[ei] * dropout_norm_mult;

                auto start_offset = dataset.index.offsets[ei] - batch_start_offset;
                auto end_offset = dataset.index.offsets[ei+1] - batch_start_offset;

                auto feature_count = end_offset - start_offset;
                auto interaction_count = feature_count * (feature_count + 1) / 2;

                fill_mask_rand(mask, (interaction_count + 63) / 64, dropout_prob_log);

                ffm_float t = predict(batch_features_data + start_offset, batch_features_data + end_offset, norm, mask);
                ffm_float expnyt = exp(-y*t);

                loss += log(1+expnyt);

                update(batch_features_data + start_offset, batch_features_data + end_offset, norm, -y * expnyt / (1+expnyt), mask);
            }
        }

        cnt += batch_end_index - batch_start_index;
    }

    clock_t end = clock();
    double elapsed = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << cnt << " examples processed in " << elapsed << " seconds, loss = " << (loss / cnt) << std::endl;

    return loss;
}


ffm_double compute_ap(const std::vector<ffm_float> & predictions, ffm_uint begin_idx, ffm_uint end_idx, ffm_uint positive_idx) {
    ffm_uint rank = 0;

    for (ffm_uint j = begin_idx; j < end_idx; ++ j)
        if (predictions[j] >= predictions[positive_idx])
            rank ++;

    if (rank > 0 && rank <= 12)
        return 1.0 / rank;
    else
        return 0.0;
}


ffm_double compute_map(const ffm_index & index, const std::vector<ffm_float> & predictions) {
    ffm_double total = 0.0;
    ffm_uint count = 0;

    ffm_uint cur_group = 0;
    ffm_uint cur_group_start_idx = 0;

    ffm_uint positive_idx = index.size + 1;

    for (ffm_uint i = 0; i < index.size; ++ i) {
        if (index.groups[i] < cur_group)
            throw std::logic_error("Groups must be ordered!");

        if (index.groups[i] > cur_group) {
            if (cur_group > 0) {
                total += compute_ap(predictions, cur_group_start_idx, i, positive_idx);
                count ++;
            }

            cur_group = index.groups[i];
            cur_group_start_idx = i;
            positive_idx = index.size + 1;
        }

        if (index.labels[i] > 0)
            positive_idx = i;
    }

    total += compute_ap(predictions, cur_group_start_idx, index.size, positive_idx);
    count ++;

    return total / count;
}


ffm_double evaluate_on_dataset(const ffm_dataset & dataset) {
    clock_t begin = clock();

    std::cout << "  Evaluating... ";
    std::cout.flush();

    auto batches = generate_batches(dataset.index, false);

    uint64_t mask[dropout_mask_size];
    fill_mask_ones(mask, dropout_mask_size);

    ffm_double loss = 0.0;
    ffm_uint cnt = 0;

    std::vector<ffm_float> predictions(dataset.index.size);

    // Iterate over batches, read each and then iterate over examples
    #pragma omp parallel for schedule(dynamic, 1) reduction(+: loss) reduction(+: cnt)
    for (ffm_uint bi = 0; bi < batches.size(); ++ bi) {
        auto batch_start_index = batches[bi].first;
        auto batch_end_index = batches[bi].second;

        auto batch_start_offset = dataset.index.offsets[batch_start_index];
        auto batch_end_offset = dataset.index.offsets[batch_end_index];

        std::vector<ffm_feature> batch_features = ffm_read_batch(dataset.data_file_name, batch_start_offset, batch_end_offset);
        ffm_feature * batch_features_data = batch_features.data();

        for (auto ei = batch_start_index; ei < batch_end_index; ++ ei) {
            ffm_float y = dataset.index.labels[ei];
            ffm_float norm = dataset.index.norms[ei];

            auto start_offset = dataset.index.offsets[ei] - batch_start_offset;
            auto end_offset = dataset.index.offsets[ei+1] - batch_start_offset;

            ffm_float t = predict(batch_features_data + start_offset, batch_features_data + end_offset, norm, mask);
            ffm_float expnyt = exp(-y*t);

            loss += log(1+expnyt);
            predictions[ei] = 1 / (1+exp(-t));
        }

        cnt += batch_end_index - batch_start_index;
    }

    // Compute map metric
    ffm_double map = compute_map(dataset.index, predictions);

    clock_t end = clock();
    double elapsed = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << cnt << " examples processed in " << elapsed << " seconds, loss = " << (loss / cnt) << ", map = " << map << std::endl;

    return loss;
}


void predict_on_dataset(const ffm_dataset & dataset, std::ostream & out) {
    clock_t begin = clock();

    std::cout << "  Predicting... ";
    std::cout.flush();

    auto batches = generate_batches(dataset.index, false);

    uint64_t mask[dropout_mask_size];
    fill_mask_ones(mask, dropout_mask_size);

    ffm_ulong cnt = 0;

    // Iterate over batches, read each and then iterate over examples
    for (ffm_ulong bi = 0; bi < batches.size(); ++ bi) {
        auto batch_start_index = batches[bi].first;
        auto batch_end_index = batches[bi].second;

        auto batch_start_offset = dataset.index.offsets[batch_start_index];
        auto batch_end_offset = dataset.index.offsets[batch_end_index];

        std::vector<ffm_feature> batch_features = ffm_read_batch(dataset.data_file_name, batch_start_offset, batch_end_offset);
        ffm_feature * batch_features_data = batch_features.data();

        for (auto ei = batch_start_index; ei < batch_end_index; ++ ei) {
            ffm_float norm = dataset.index.norms[ei];

            auto start_offset = dataset.index.offsets[ei] - batch_start_offset;
            auto end_offset = dataset.index.offsets[ei+1] - batch_start_offset;

            ffm_float t = predict(batch_features_data + start_offset, batch_features_data + end_offset, norm, mask);

            out << 1/(1+exp(-t)) << std::endl;
        }

        cnt += batch_end_index - batch_start_index;
    }

    clock_t end = clock();
    double elapsed = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << cnt << " examples processed in " << elapsed << " seconds" << std::endl;
}


template <typename D>
void init_weights(ffm_float * weights, ffm_uint n, D gen) {
    ffm_float * w = weights;

    for(ffm_uint i = 0; i < n; i++) {
        for (ffm_uint d = 0; d < n_dim; d++, w++)
            *w = gen(rnd);

        for (ffm_uint d = n_dim; d < n_dim_aligned; d++, w++)
            *w = 0;

        for (ffm_uint d = n_dim_aligned; d < 2*n_dim_aligned; d++, w++)
            *w = 1;
    }
}


void init_model() {
    weights = malloc_aligned_float(n_features * n_fields * n_dim_aligned * 2);

    init_weights(weights, n_features * n_fields, std::uniform_real_distribution<ffm_float>(0.0, 1.0/sqrt(n_dim)));
}


ffm_dataset open_dataset(const std::string & file_name) {
    ffm_dataset res;

    std::cout << "Loading " << file_name << ".index... ";
    std::cout.flush();

    res.index = ffm_read_index(file_name + ".index");
    res.data_file_name = file_name + ".data";

    std::cout << res.index.size << " examples" << std::endl;

    return res;
}

class program_options {
    boost::program_options::options_description desc;
public:
    std::string train_file_name;
    std::string val_file_name;
    std::string test_file_name;
    std::string pred_file_name;

    ffm_uint n_epochs;
    ffm_uint n_threads;

    bool restricted;
public:
    program_options(int ac, char* av[]): desc("Allowed options"), n_epochs(10), n_threads(4) {
        using namespace boost::program_options;

        desc.add_options()
            ("help", "train dataset file")
            ("train", value<std::string>(&train_file_name)->required(), "train dataset file")
            ("val", value<std::string>(&val_file_name), "validation dataset file")
            ("test", value<std::string>(&test_file_name), "test dataset file")
            ("pred", value<std::string>(&pred_file_name), "file to save predictions")
            ("epochs", value<ffm_uint>(&n_epochs), "number of epochs (default 10)")
            ("threads", value<ffm_uint>(&n_threads), "number of threads (default 4)")
            ("restricted", "restrict feature interactions to (E+C) * (C+A)")
        ;

        variables_map vm;
        store(parse_command_line(ac, av, desc), vm);

        if (vm.count("help") > 0) {
            std::cout << desc << std::endl;
            exit(0);
        }

        restricted = vm.count("restricted") > 0;

        notify(vm);
    }
};


int main(int ac, char* av[]) {
    using namespace std;

    program_options opts(ac, av);

    omp_set_num_threads(opts.n_threads);

    // Restricted FFM, exclude interactions of Event * Event or Ad * Ad features
    if (opts.restricted) {
        max_b_field = 19;
        min_a_field = 10;
    }

    init_model();

    if (opts.val_file_name.empty()) { // No validation set given, just train
        auto ds_train = open_dataset(opts.train_file_name);

        for (ffm_uint epoch = 0; epoch < opts.n_epochs; ++ epoch) {
            cout << "Epoch " << epoch << "..." << endl;

            train_on_dataset(ds_train);
        }
    } else { // Train with validation each epoch
        auto ds_train = open_dataset(opts.train_file_name);
        auto ds_val = open_dataset(opts.val_file_name);

        for (ffm_uint epoch = 0; epoch < opts.n_epochs; ++ epoch) {
            cout << "Epoch " << epoch << "..." << endl;

            train_on_dataset(ds_train);
            evaluate_on_dataset(ds_val);
        }
    }

    if (!opts.test_file_name.empty() && !opts.pred_file_name.empty()) {
        auto ds_test = open_dataset(opts.test_file_name);

        ofstream out(opts.pred_file_name);
        predict_on_dataset(ds_test, out);
    }

    cout << "Done." << endl;
}
