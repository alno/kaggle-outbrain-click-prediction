#pragma once

#include <string>
#include <vector>

typedef void * ffm_file;

typedef uint32_t ffm_uint;
typedef uint64_t ffm_ulong;
typedef float ffm_float;
typedef double ffm_double;

struct ffm_feature {
    ffm_uint field;
    ffm_uint index;
    ffm_float value;
};

// Structure for fast access to data
struct ffm_index {
    ffm_ulong size; // Number of examples;

    std::vector<ffm_float> labels; // Target values of examples (size N)
    std::vector<ffm_ulong> offsets; // Offsets of example data (size N +1) in number of features
    std::vector<ffm_float> norms; // Squares of l2 norm of examples (size N)
};

// IO functions

void ffm_write_index(const std::string & file_name, const ffm_index & index);

ffm_index ffm_read_index(const std::string & file_name);

std::vector<ffm_feature> ffm_read_batch(const std::string & file_name, ffm_ulong from, ffm_ulong to);

// Writes data files in sequential order
class ffm_stream_data_writer {
    ffm_file file;
    ffm_ulong offset;
public:
    ffm_stream_data_writer(const std::string & file_name);
    ~ffm_stream_data_writer();

    ffm_ulong write(const std::vector<ffm_feature> & features);
};
