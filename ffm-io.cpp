#include <stdexcept>

#include <cstdio>

#include "ffm.h"


// index file reading / writing


void ffm_write_index(const std::string & file_name, const ffm_index & index) {
    using namespace std;

    if (index.labels.size() != index.size)
        throw runtime_error("Invalid index labels size");

    if (index.offsets.size() != index.size + 1)
        throw runtime_error("Invalid index offsets size");

    if (index.norms.size() != index.size)
        throw runtime_error("Invalid index norms size");

    if (index.groups.size() != index.size)
        throw runtime_error("Invalid index groups size");

    FILE * file = fopen(file_name.c_str(), "wb");

    if(file == nullptr)
        throw runtime_error(string("Can't open index file ") + file_name);

    if (fwrite(&index.size, sizeof(ffm_ulong), 1, file) != 1)
        throw runtime_error("Error writing example count");

    if (fwrite(index.labels.data(), sizeof(ffm_float), index.labels.size(), file) != index.labels.size())
        throw runtime_error("Error writing labels");

    if (fwrite(index.offsets.data(), sizeof(ffm_ulong), index.offsets.size(), file) != index.offsets.size())
        throw runtime_error("Error writing offsets");

    if (fwrite(index.norms.data(), sizeof(ffm_float), index.norms.size(), file) != index.norms.size())
        throw runtime_error("Error writing norms");

    if (fwrite(index.groups.data(), sizeof(ffm_uint), index.groups.size(), file) != index.groups.size())
        throw runtime_error("Error writing groups");

    fclose(file);
}

ffm_index ffm_read_index(const std::string & file_name) {
    using namespace std;

    ffm_index index;
    FILE * file = fopen(file_name.c_str(), "rb");

    if(file == nullptr)
        throw runtime_error(string("Can't open index file ") + file_name);

    if (fread(&index.size, sizeof(ffm_ulong), 1, file) != 1)
        throw runtime_error("Error reading example count");

    // Reserve space for y and offsets
    index.labels.resize(index.size, 0);
    index.offsets.resize(index.size + 1, 0);
    index.norms.resize(index.size, 0);
    index.groups.resize(index.size, 0);

    if (fread(index.labels.data(), sizeof(ffm_float), index.labels.size(), file) != index.labels.size())
        throw runtime_error("Error reading labels");

    if (fread(index.offsets.data(), sizeof(ffm_ulong), index.offsets.size(), file) != index.offsets.size())
        throw runtime_error("Error reading offsets");

    if (fread(index.norms.data(), sizeof(ffm_float), index.norms.size(), file) != index.norms.size())
        throw runtime_error("Error reading norms");

    if (fread(index.groups.data(), sizeof(ffm_uint), index.groups.size(), file) != index.groups.size())
        throw runtime_error("Error reading groups");

    fclose(file);

    return index;
}

// batch data reading

std::vector<ffm_feature> ffm_read_batch(const std::string & file_name, ffm_ulong from, ffm_ulong to) {
    std::vector<ffm_feature> features(to - from);
    ffm_read_batch(file_name, from, to, features);
    return features;
};

void ffm_read_batch(const std::string & file_name, ffm_ulong from, ffm_ulong to, std::vector<ffm_feature> & features) {
    using namespace std;

    if (to < from)
        throw runtime_error("Wrong range");

    features.resize(to - from);

    // Empty range, no need to read
    if (to == from)
        return;

    FILE * file = fopen(file_name.c_str(), "rb");

    if (file == nullptr)
        throw runtime_error(string("Can't open data file ") + file_name);

    if (fseek((FILE *)file, from * sizeof(ffm_feature), SEEK_SET) != 0)
        throw new runtime_error("Can't set file pos");

    if (fread(features.data(), sizeof(ffm_feature), features.size(), (FILE *)file) != features.size())
        throw new runtime_error("Can't read data");

    fclose(file);
}

// stream data writing

ffm_stream_data_writer::ffm_stream_data_writer(const std::string & file_name): offset(0) {
    using namespace std;

    file = fopen(file_name.c_str(), "wb");

    if(file == nullptr)
        throw runtime_error(string("Can't open data file ") + file_name);
}

ffm_stream_data_writer::~ffm_stream_data_writer() {
    fclose((FILE *)file);
}

ffm_ulong ffm_stream_data_writer::write(const std::vector<ffm_feature> & features) {
    if (fwrite(features.data(), sizeof(ffm_feature), features.size(), (FILE *)file) != features.size())
        throw std::runtime_error("Error writing example count");

    offset += features.size();

    return offset;
}
