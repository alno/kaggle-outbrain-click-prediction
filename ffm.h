#pragma once

#include <string>
#include <vector>
#include <functional>

typedef void * ffm_file;

typedef uint32_t ffm_uint;
typedef uint64_t ffm_ulong;
typedef float ffm_float;
typedef double ffm_double;


const ffm_uint ffm_hash_bits = 20;
const ffm_uint ffm_hash_mask = (1 << ffm_hash_bits) - 1;


struct ffm_feature {
    ffm_uint index;
    ffm_float value;
};

// Structure for fast access to data
struct ffm_index {
    ffm_ulong size; // Number of examples;

    std::vector<ffm_float> labels; // Target values of examples (size N)
    std::vector<ffm_ulong> offsets; // Offsets of example data (size N +1) in number of features
    std::vector<ffm_float> norms; // Squares of l2 norm of examples (size N)
    std::vector<ffm_uint> groups; // Group identifiers for MAP calculation
};

// IO functions

void ffm_write_index(const std::string & file_name, const ffm_index & index);

ffm_index ffm_read_index(const std::string & file_name);

std::vector<ffm_feature> ffm_read_batch(const std::string & file_name, ffm_ulong from, ffm_ulong to);
void ffm_read_batch(const std::string & file_name, ffm_ulong from, ffm_ulong to, std::vector<ffm_feature> & features);


// Writes data files in sequential order
class ffm_stream_data_writer {
    ffm_file file;
    ffm_ulong offset;
public:
    ffm_stream_data_writer(const std::string & file_name);
    ~ffm_stream_data_writer();

    ffm_ulong write(const std::vector<ffm_feature> & features);
};

// Feature builder helper


class ffm_feature_vector_builder {
    std::vector<ffm_feature> vector;
    std::hash<std::string> str_hash;

    uint32_t hash_offset, hash_size;
public:
    ffm_feature_vector_builder(uint32_t hash_offset): hash_offset(hash_offset), hash_size((1 << ffm_hash_bits) - hash_offset) {}

    void raw(uint32_t field, uint32_t index, float value = 1.0) {
        ffm_feature f;
        f.index = (field << ffm_hash_bits) | (index & ffm_hash_mask);
        f.value = value;

        vector.push_back(f);
    }

    void hashed(uint32_t field, uint32_t category, float value = 1.0) {
        raw(field, h(category + field * 2654435761) % hash_size + hash_offset, value);
    }

    void hashed(uint32_t field, const std::string & category, float value = 1.0) {
        raw(field, h(str_hash(category) + field * 2654435761) % hash_size + hash_offset, value);
    }

    const std::vector<ffm_feature> & data() {
        return vector;
    }

    ffm_float norm() {
        ffm_float norm = 0.0;

        for (auto fi = vector.begin(); fi != vector.end(); ++ fi)
            norm += fi->value * fi->value;

        return norm;
    }
private:
    uint32_t h(uint32_t a) {
        a = (a ^ 61) ^ (a >> 16);
        a = a + (a << 3);
        a = a ^ (a >> 4);
        a = a * 0x27d4eb2d;
        a = a ^ (a >> 15);
        return a;
    }
};
