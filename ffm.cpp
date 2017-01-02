#include "ffm.h"

#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>

#include <pmmintrin.h>
#include <omp.h>

#include <boost/program_options.hpp>

ffm_uint batch_size = 500000;
ffm_uint n_threads = 4;

const ffm_ulong n_features = (1 << 19) + 100;
const ffm_ulong n_fields = 50;
const ffm_ulong n_dim = 4;


ffm_float weights[n_features * n_fields * n_dim * 2];

const ffm_ulong index_stride = n_fields * n_dim * 2;
const ffm_ulong field_stride = n_dim * 2;

std::default_random_engine rnd(1234);

ffm_float norm = 1.0;

ffm_float eta = 0.2;
ffm_float lambda = 0.0004;


template <typename T>
T min(T a, T b) {
    return a < b ? a : b;
}

struct ffm_dataset {
    ffm_index index;
    std::string data_file_name;
};


ffm_float predict(const ffm_feature * start, const ffm_feature * end) {
    __m128 xmm_total = _mm_setzero_ps();

    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        ffm_uint field_a = fa->field;
        ffm_uint index_a = fa->index;
        ffm_float value_a = fa->value / norm;

        for (const ffm_feature * fb = start + 1; fb != end; ++ fb) {
            ffm_uint field_b = fb->field;
            ffm_uint index_b = fb->index;
            ffm_float value_b = fb->value / norm;

            ffm_float * wa = weights + index_a * index_stride + field_b * field_stride;
            ffm_float * wb = weights + index_b * index_stride + field_a * field_stride;

            __m128 xmm_val = _mm_set1_ps(value_a * value_b);

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


void update(const ffm_feature * start, const ffm_feature * end, ffm_float kappa) {
    __m128 xmm_eta = _mm_set1_ps(eta);
    __m128 xmm_lambda = _mm_set1_ps(lambda);

    for (const ffm_feature * fa = start; fa != end; ++ fa) {
        ffm_uint field_a = fa->field;
        ffm_uint index_a = fa->index;
        ffm_float value_a = fa->value / norm;

        for (const ffm_feature * fb = start + 1; fb != end; ++ fb) {
            ffm_uint field_b = fb->field;
            ffm_uint index_b = fb->index;
            ffm_float value_b = fb->value / norm;

            ffm_float * wa = weights + index_a * index_stride + field_b * field_stride;
            ffm_float * wb = weights + index_b * index_stride + field_a * field_stride;

            ffm_float * wga = wa + n_dim;
            ffm_float * wgb = wb + n_dim;

            __m128 xmm_kappa_val = _mm_set1_ps(kappa * value_a * value_b);

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
}


std::vector<std::pair<ffm_ulong, ffm_ulong>> generate_batches(const ffm_index & index, bool shuffle) {
    std::vector<std::pair<ffm_ulong, ffm_ulong>> batches;

    for (ffm_ulong batch_start = 0; batch_start < index.size; batch_start += batch_size)
        batches.push_back(std::make_pair(batch_start, min(batch_start + batch_size, index.size)));

    if (shuffle)
        std::shuffle(batches.begin(), batches.end(), rnd);

    return batches;
}


ffm_float train_on_dataset(const ffm_dataset & dataset) {
    clock_t begin = clock();

    std::cout << "  Training... ";
    std::cout.flush();

    auto batches = generate_batches(dataset.index, true);

    ffm_float loss = 0.0f;
    ffm_ulong cnt = 0;

    // Iterate over batches, read each and then iterate over examples
    #pragma omp parallel for schedule(dynamic, 1) reduction(+: loss) reduction(+: cnt)
    for (ffm_ulong bi = 0; bi < batches.size(); ++ bi) {
        auto batch_start_index = batches[bi].first;
        auto batch_end_index = batches[bi].second;

        auto batch_start_offset = dataset.index.offsets[batch_start_index];
        auto batch_end_offset = dataset.index.offsets[batch_end_index];

        std::vector<ffm_feature> batch_features = ffm_read_batch(dataset.data_file_name, batch_start_offset, batch_end_offset);
        ffm_feature * batch_features_data = batch_features.data();

        for (auto ei = batch_start_index; ei < batch_end_index; ++ ei) {
            ffm_float y = dataset.index.labels[ei];

            auto start_offset = dataset.index.offsets[ei] - batch_start_offset;
            auto end_offset = dataset.index.offsets[ei+1] - batch_start_offset;

            ffm_float t = predict(batch_features_data + start_offset, batch_features_data + end_offset);
            ffm_float expnyt = exp(-y*t);

            loss += log(1+expnyt);

            update(batch_features_data + start_offset, batch_features_data + end_offset, -y * expnyt / (1+expnyt));
        }

        cnt += batch_end_index - batch_start_index;
    }

    clock_t end = clock();
    double elapsed = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << cnt << " examples processed in " << elapsed << " seconds, loss = " << (loss / cnt) << std::endl;

    return loss;
}


ffm_float evaluate_on_dataset(const ffm_dataset & dataset) {
    clock_t begin = clock();

    std::cout << "  Evaluating... ";
    std::cout.flush();

    auto batches = generate_batches(dataset.index, false);

    ffm_float loss = 0.0f;
    ffm_ulong cnt = 0;

    // Iterate over batches, read each and then iterate over examples
    #pragma omp parallel for schedule(dynamic, 1) reduction(+: loss) reduction(+: cnt)
    for (ffm_ulong bi = 0; bi < batches.size(); ++ bi) {
        auto batch_start_index = batches[bi].first;
        auto batch_end_index = batches[bi].second;

        auto batch_start_offset = dataset.index.offsets[batch_start_index];
        auto batch_end_offset = dataset.index.offsets[batch_end_index];

        std::vector<ffm_feature> batch_features = ffm_read_batch(dataset.data_file_name, batch_start_offset, batch_end_offset);
        ffm_feature * batch_features_data = batch_features.data();

        for (auto ei = batch_start_index; ei < batch_end_index; ++ ei) {
            ffm_float y = dataset.index.labels[ei];

            auto start_offset = dataset.index.offsets[ei] - batch_start_offset;
            auto end_offset = dataset.index.offsets[ei+1] - batch_start_offset;

            ffm_float t = predict(batch_features_data + start_offset, batch_features_data + end_offset);
            ffm_float expnyt = exp(-y*t);

            loss += log(1+expnyt);
        }

        cnt += batch_end_index - batch_start_index;
    }

    clock_t end = clock();
    double elapsed = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << cnt << " examples processed in " << elapsed << " seconds, loss = " << (loss / cnt) << std::endl;

    return loss;
}


void predict_on_dataset(const ffm_dataset & dataset, std::ostream & out) {
    clock_t begin = clock();

    std::cout << "  Predicting... ";
    std::cout.flush();

    auto batches = generate_batches(dataset.index, false);

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
            auto start_offset = dataset.index.offsets[ei] - batch_start_offset;
            auto end_offset = dataset.index.offsets[ei+1] - batch_start_offset;

            ffm_float t = predict(batch_features_data + start_offset, batch_features_data + end_offset);

            out << 1/(1+exp(-t)) << std::endl;
        }

        cnt += batch_end_index - batch_start_index;
    }

    clock_t end = clock();
    double elapsed = double(end - begin) / CLOCKS_PER_SEC;

    std::cout << cnt << " examples processed in " << elapsed << " seconds" << std::endl;
}


void init_model() {
    std::uniform_real_distribution<ffm_float> distribution(0.0, 1.0);

    ffm_float * w = weights;
    ffm_float coef = 1.0f / sqrt(n_dim);

    for(ffm_uint j = 0; j < n_features; j++) {
        for(ffm_uint f = 0; f < n_fields; f++) {
            for(ffm_uint d = 0; d < n_dim; d++, w++)
                *w = coef * distribution(rnd);

            for(ffm_uint d = n_dim; d < 2*n_dim; d++, w++)
                *w = 1;
        }
    }
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
public:
    program_options(int ac, char* av[]): desc("Allowed options"), n_epochs(1) {
        using namespace boost::program_options;

        desc.add_options()
            ("help", "train dataset file")
            ("train", value<std::string>(&train_file_name)->required(), "train dataset file")
            ("val", value<std::string>(&val_file_name), "validation dataset file")
            ("test", value<std::string>(&test_file_name), "test dataset file")
            ("pred", value<std::string>(&pred_file_name), "file to save predictions")
            ("epochs", value<ffm_uint>(&n_epochs), "number of epochs")
        ;

        variables_map vm;
        store(parse_command_line(ac, av, desc), vm);

        if (vm.count("help") > 0) {
            std::cout << desc << std::endl;
            exit(0);
        }

        notify(vm);
    }
};


int main(int ac, char* av[]) {
    using namespace std;

    program_options opts(ac, av);

    omp_set_num_threads(n_threads);

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
