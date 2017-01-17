#include "ffm.h"

#include "ffm-model.h"
#include "ffm-nn-model.h"
#include "ftrl-model.h"
#include "nn-model.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <algorithm>

#include <immintrin.h>
#include <omp.h>

#include <boost/program_options.hpp>

// Batch configuration
const ffm_uint batch_size = 20000;
const ffm_uint mini_batch_size = 24;

// Dropout configuration
const ffm_uint dropout_mask_max_size = 4000; // in 64-bit words


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

    return batches;
}


void fill_mask_rand(uint64_t * mask, int size, int zero_prob_log) {
    memset(mask, 0, size * sizeof(uint64_t));

    for (int p = 0; p < size; ++ p) {
        for (int i = 0; i < zero_prob_log; ++ i) {
            long long unsigned int v;
            if (_rdrand64_step(&v) != 1)
                throw std::runtime_error("Error generating random number!");

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
ffm_double train_on_dataset(const std::vector<M*> & models, const ffm_dataset & dataset, uint dropout_prob_log) {
    float dropout_mult = (1 << dropout_prob_log) / ((1 << dropout_prob_log) - 1.0f);

    time_t start_time = time(nullptr);

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

        uint64_t dropout_mask[dropout_mask_max_size];

        std::vector<float> ts(batch_end_index - batch_start_index);
        std::vector<uint> tc(batch_end_index - batch_start_index);


        for (uint mi = 0; mi < models.size(); ++ mi) {
            std::shuffle(mini_batches.begin(), mini_batches.end(), rnd);

            for (auto mb = mini_batches.begin(); mb != mini_batches.end(); ++ mb) {
                for (auto ei = mb->first; ei < mb->second; ++ ei) {
                    ffm_float y = dataset.index.labels[ei];
                    ffm_float norm = dataset.index.norms[ei];

                    auto start_offset = dataset.index.offsets[ei] - batch_start_offset;
                    auto end_offset = dataset.index.offsets[ei+1] - batch_start_offset;

                    auto dropout_mask_size = models[mi]->get_dropout_mask_size(batch_features_data + start_offset, batch_features_data + end_offset);

                    fill_mask_rand(dropout_mask, (dropout_mask_size + 63) / 64, dropout_prob_log);

                    float t = models[mi]->predict(batch_features_data + start_offset, batch_features_data + end_offset, norm, dropout_mask, dropout_mult);
                    float expnyt = exp(-y*t);

                    models[mi]->update(batch_features_data + start_offset, batch_features_data + end_offset, norm, -y * expnyt / (1+expnyt), dropout_mask, dropout_mult);

                    uint i = ei - batch_start_index;
                    ts[i] += t;
                    tc[i] ++;
                }
            }
        }

        for (uint i = 0; i < batch_end_index - batch_start_index; ++ i)
            loss += log(1+exp(-dataset.index.labels[i + batch_start_index]*ts[i]/tc[i]));

        cnt += batch_end_index - batch_start_index;
    }

    std::cout << cnt << " examples processed in " << (time(nullptr) - start_time) << " seconds, loss = " << std::fixed << std::setprecision(5) << (loss / cnt) << std::endl;

    return loss;
}


template <typename M>
ffm_double evaluate_on_dataset(const std::vector<M*> & models, const ffm_dataset & dataset) {
    time_t start_time = time(nullptr);

    std::cout << "  Evaluating... ";
    std::cout.flush();

    auto batches = generate_batches(dataset.index, false);

    uint64_t dropout_mask[dropout_mask_max_size];
    fill_mask_ones(dropout_mask, dropout_mask_max_size);

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

            float ts = 0.0;
            uint tc = 0;

            for (uint mi = 0; mi < models.size(); ++ mi) {
                ts += models[mi]->predict(batch_features_data + start_offset, batch_features_data + end_offset, norm, dropout_mask, 1);
                tc ++;
            }

            loss += log(1+exp(-y*ts/tc));
            predictions[ei] = 1 / (1+exp(-ts/tc));
        }

        cnt += batch_end_index - batch_start_index;
    }

    // Compute map metric
    ffm_double map = compute_map(dataset.index, predictions);

    std::cout << cnt << " examples processed in " << (time(nullptr) - start_time) << " seconds, loss = " << std::fixed << std::setprecision(5) << (loss / cnt) << ", map = " << map << std::endl;

    return loss;
}

template <typename M>
void predict_on_dataset(const std::vector<M*> & models, const ffm_dataset & dataset, std::ostream & out) {
    time_t start_time = time(nullptr);

    std::cout << "  Predicting... ";
    std::cout.flush();

    auto batches = generate_batches(dataset.index, false);

    uint64_t dropout_mask[dropout_mask_max_size];
    fill_mask_ones(dropout_mask, dropout_mask_max_size);

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

            float ts = 0.0;
            uint tc = 0;

            for (uint mi = 0; mi < models.size(); ++ mi) {
                ts += models[mi]->predict(batch_features_data + start_offset, batch_features_data + end_offset, norm, dropout_mask, 1);
                tc ++;
            }

            out << 1/(1+exp(-ts/tc)) << std::endl;
        }

        cnt += batch_end_index - batch_start_index;
    }

    std::cout << cnt << " examples processed in " << (time(nullptr) - start_time) << " seconds" << std::endl;
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

    uint n_epochs;
    uint n_threads;
    uint n_models;
    uint seed;

    bool restricted;

    uint dropout_prob_log;

    float eta, lambda;
public:
    program_options(int ac, char* av[]):
        desc("Allowed options"), model_name("ffm"), n_epochs(10), n_threads(4), n_models(1), seed(2017),
        dropout_prob_log(1), eta(0), lambda(0)
    {
        using namespace boost::program_options;

        desc.add_options()
            ("help", "train dataset file")
            ("model", value<std::string>(&model_name), "model name")
            ("train", value<std::string>(&train_file_name)->required(), "train dataset file")
            ("val", value<std::string>(&val_file_name), "validation dataset file")
            ("test", value<std::string>(&test_file_name), "test dataset file")
            ("pred", value<std::string>(&pred_file_name), "file to save predictions")
            ("epochs", value<uint>(&n_epochs), "number of epochs (default 10)")
            ("threads", value<uint>(&n_threads), "number of threads (default 4)")
            ("average", value<uint>(&n_models), "number of models to average (default 1)")
            ("seed", value<uint>(&seed), "random seed")
            ("dropout-log", value<uint>(&dropout_prob_log), "binary log of dropout probability (default 1)")
            ("eta", value<float>(&eta), "learning rate")
            ("lambda", value<float>(&lambda), "l2 regularization coeff")
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
void apply(const std::vector<M*> & models, program_options & opts) {
    using namespace std;

    if (opts.val_file_name.empty()) { // No validation set given, just train
        auto ds_train = open_dataset(opts.train_file_name);

        for (ffm_uint epoch = 0; epoch < opts.n_epochs; ++ epoch) {
            cout << "Epoch " << epoch << "..." << endl;

            train_on_dataset(models, ds_train, opts.dropout_prob_log);
        }
    } else { // Train with validation each epoch
        auto ds_train = open_dataset(opts.train_file_name);
        auto ds_val = open_dataset(opts.val_file_name);

        for (ffm_uint epoch = 0; epoch < opts.n_epochs; ++ epoch) {
            cout << "Epoch " << epoch << "..." << endl;

            train_on_dataset(models, ds_train, opts.dropout_prob_log);
            evaluate_on_dataset(models, ds_val);
        }
    }

    if (!opts.test_file_name.empty() && !opts.pred_file_name.empty()) {
        auto ds_test = open_dataset(opts.test_file_name);

        ofstream out(opts.pred_file_name);
        predict_on_dataset(models, ds_test, out);
    }
}


int main(int ac, char* av[]) {
    program_options opts(ac, av);

    // Init global state
    omp_set_num_threads(opts.n_threads);
    rnd.seed(opts.seed);

    // Run model
    if (opts.model_name == "ffm") {
        float eta = opts.eta > 0 ? opts.eta : 0.2;
        float lambda = opts.lambda > 0 ? opts.lambda : 0.00002;

        std::vector<ffm_model*> models;

        for (uint i = 0; i < opts.n_models; ++ i)
            models.push_back(new ffm_model(opts.seed + 100 + i * 17, opts.restricted, eta, lambda));

        apply(models, opts);
    } else if (opts.model_name == "ffm-nn") {
        float eta = opts.eta > 0 ? opts.eta : 0.05;
        float lambda = opts.lambda > 0 ? opts.lambda : 0.00002;

        std::vector<ffm_nn_model*> models;

        for (uint i = 0; i < opts.n_models; ++ i)
            models.push_back(new ffm_nn_model(opts.seed + 100 + i * 17, opts.restricted, eta, lambda, 0.0001));

        apply(models, opts);
    } else if (opts.model_name == "ftrl") {
        std::vector<ftrl_model*> models;

        for (uint i = 0; i < opts.n_models; ++ i)
            models.push_back(new ftrl_model(24, 1.0, 2.0, 2e-4, 5e-4));

        apply(models, opts);
    } else if (opts.model_name == "nn") {
        float eta = opts.eta > 0 ? opts.eta : 0.02;
        float lambda = opts.lambda > 0 ? opts.lambda : 0.00002;

        std::vector<nn_model*> models;

        for (uint i = 0; i < opts.n_models; ++ i)
            models.push_back(new nn_model(opts.seed + 100 + i * 17, eta, lambda));

        apply(models, opts);
    } else {
        throw std::runtime_error(std::string("Unknown model ") + opts.model_name);
    }

    std::cout << "Done." << std::endl;
}
