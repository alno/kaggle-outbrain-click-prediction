#include "ffm.h"

#include "ffm-model.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <algorithm>

#include <immintrin.h>
#include <omp.h>

#include <boost/program_options.hpp>

// Batch configuration
const ffm_uint batch_size = 10000;
const ffm_uint mini_batch_size = 32;

// Dropout configuration
const ffm_uint dropout_mask_size = 100; // in 64-bit words
const ffm_uint dropout_prob_log = 1; // 0.5 dropout rate
const ffm_float dropout_norm_mult = ((1 << dropout_prob_log) - 1.0f) / (1 << dropout_prob_log);

std::default_random_engine rnd(2017);


template <typename T>
T min(T a, T b) {
    return a < b ? a : b;
}

struct ffm_dataset {
    ffm_index index;
    std::string data_file_name;
};



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


template <typename M>
ffm_double train_on_dataset(M & model, const ffm_dataset & dataset) {
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

                ffm_float t = model.predict(batch_features_data + start_offset, batch_features_data + end_offset, norm, mask);
                ffm_float expnyt = exp(-y*t);

                loss += log(1+expnyt);

                model.update(batch_features_data + start_offset, batch_features_data + end_offset, norm, -y * expnyt / (1+expnyt), mask);
            }
        }

        cnt += batch_end_index - batch_start_index;
    }

    clock_t end = clock();
    int elapsed = (end - begin) / CLOCKS_PER_SEC;

    std::cout << cnt << " examples processed in " << elapsed << " seconds, loss = " << std::fixed << std::setprecision(5) << (loss / cnt) << std::endl;

    return loss;
}


template <typename M>
ffm_double evaluate_on_dataset(M & model, const ffm_dataset & dataset) {
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

            ffm_float t = model.predict(batch_features_data + start_offset, batch_features_data + end_offset, norm, mask);
            ffm_float expnyt = exp(-y*t);

            loss += log(1+expnyt);
            predictions[ei] = 1 / (1+exp(-t));
        }

        cnt += batch_end_index - batch_start_index;
    }

    // Compute map metric
    ffm_double map = compute_map(dataset.index, predictions);

    clock_t end = clock();
    int elapsed = (end - begin) / CLOCKS_PER_SEC;

    std::cout << cnt << " examples processed in " << elapsed << " seconds, loss = " << std::fixed << std::setprecision(5) << (loss / cnt) << ", map = " << map << std::endl;

    return loss;
}

template <typename M>
void predict_on_dataset(M & model, const ffm_dataset & dataset, std::ostream & out) {
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

            ffm_float t = model.predict(batch_features_data + start_offset, batch_features_data + end_offset, norm, mask);

            out << 1/(1+exp(-t)) << std::endl;
        }

        cnt += batch_end_index - batch_start_index;
    }

    clock_t end = clock();
    int elapsed = (end - begin) / CLOCKS_PER_SEC;

    std::cout << cnt << " examples processed in " << elapsed << " seconds" << std::endl;
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

    std::string model_name;

    ffm_uint n_epochs;
    ffm_uint n_threads;

    bool restricted;
public:
    program_options(int ac, char* av[]): desc("Allowed options"), model_name("ffm"), n_epochs(10), n_threads(4) {
        using namespace boost::program_options;

        desc.add_options()
            ("help", "train dataset file")
            ("model", value<std::string>(&model_name), "model name")
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


template <typename M>
void apply(M & model, program_options & opts) {
    using namespace std;

    if (opts.val_file_name.empty()) { // No validation set given, just train
        auto ds_train = open_dataset(opts.train_file_name);

        for (ffm_uint epoch = 0; epoch < opts.n_epochs; ++ epoch) {
            cout << "Epoch " << epoch << "..." << endl;

            train_on_dataset(model, ds_train);
        }
    } else { // Train with validation each epoch
        auto ds_train = open_dataset(opts.train_file_name);
        auto ds_val = open_dataset(opts.val_file_name);

        for (ffm_uint epoch = 0; epoch < opts.n_epochs; ++ epoch) {
            cout << "Epoch " << epoch << "..." << endl;

            train_on_dataset(model, ds_train);
            evaluate_on_dataset(model, ds_val);
        }
    }

    if (!opts.test_file_name.empty() && !opts.pred_file_name.empty()) {
        auto ds_test = open_dataset(opts.test_file_name);

        ofstream out(opts.pred_file_name);
        predict_on_dataset(model, ds_test, out);
    }
}


int main(int ac, char* av[]) {
    program_options opts(ac, av);

    omp_set_num_threads(opts.n_threads);

    if (opts.model_name == "ffm") {
        ffm_model model(2017, opts.restricted);
        apply(model, opts);
    } else {
        throw std::runtime_error(std::string("Unknown model ") + opts.model_name);
    }

    std::cout << "Done." << std::endl;
}
