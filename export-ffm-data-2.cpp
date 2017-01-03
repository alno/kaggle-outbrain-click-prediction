#include "util/io.h"
#include "util/data.h"
#include "util/generation.h"

#include "ffm.h"

#include <functional>
#include <cmath>

std::vector<std::pair<std::vector<std::string>, std::string>> filesets = {
    { { "cache/clicks_val_train.csv.gz", "cache/leak_val_train.csv.gz", "cache/similarity_val_train.csv.gz" }, "cache/val_train_ffm_2" },
    { { "cache/clicks_val_test.csv.gz", "cache/leak_val_test.csv.gz", "cache/similarity_val_test.csv.gz" }, "cache/val_test_ffm_2" },
    { { "../input/clicks_train.csv.gz", "cache/leak_full_train.csv.gz", "cache/similarity_full_train.csv.gz" }, "cache/full_train_ffm_2" },
    { { "../input/clicks_test.csv.gz", "cache/leak_full_test.csv.gz", "cache/similarity_full_test.csv.gz" }, "cache/full_test_ffm_2" },
};

std::hash<std::string> str_hash;

uint32_t hash_offset = 100;
uint32_t hash_base = 1 << 19;

uint32_t max_index = 0;
uint32_t max_field = 0;

uint32_t h(uint32_t a, uint32_t f) {
    a = a + f * 2654435761;
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return (a % hash_base) + hash_offset;
}

uint32_t h(const std::string & a, uint32_t f) {
    return (((str_hash(a) + f) * 2654435761) % hash_base) + hash_offset;
}

ffm_feature feature_raw(uint32_t field, uint32_t index, float value = 1.0) {
    ffm_feature f;
    f.field = field;
    f.index = index;
    f.value = value;

    if (index > max_index)
        max_index = index;

    if (field > max_field)
        max_field = field;

    return f;
}

ffm_feature feature_hashed(uint32_t field, uint32_t category, float value = 1.0) {
    return feature_raw(field, h(category, field), value);
}

ffm_feature feature_hashed(uint32_t field, const std::string & category, float value = 1.0) {
    return feature_raw(field, h(category, field), value);
}

inline float pos_time_diff(int64_t td) {
    if (td < 0)
        return 0;

    return log(1 + td) / 100;
}

inline float time_diff(int64_t td) {
    if (td < 0)
        return - log(1 - td) / 100;

    return log(1 + td) / 100;
}

ffm_float norm(const std::vector<ffm_feature> & features) {
    ffm_float norm = 0.0;

    for (auto fi = features.begin(); fi != features.end(); ++ fi)
        norm += fi->value * fi->value;

    return norm;
}


class writer {
    std::string file_name;

    ffm_stream_data_writer data_out;
    ffm_index index;
public:
    writer(const std::string & file_name): file_name(file_name), data_out(file_name + ".data") {
        index.size = 0;
        index.offsets.push_back(0);
    }

    void write(const reference_data & data, const std::vector<std::vector<std::string>> & rows);
    void finish();
};


void writer::write(const reference_data & data, const std::vector<std::vector<std::string>> & rows) {
    int event_id = stoi(rows[0][0]);
    int ad_id = stoi(rows[0][1]);

    int leak_viewed = stoi(rows[1][0]);
    int leak_not_viewed = stoi(rows[1][1]);

    //

    auto ad = data.ads[ad_id];
    auto event = data.events[event_id];

    auto ad_doc = data.documents.at(ad.document_id);
    auto ad_doc_categories = data.document_categories.equal_range(ad.document_id);
    //auto ad_doc_topics = data.document_topics.equal_range(ad.document_id);
    //auto ad_doc_entities = data.document_entities.equal_range(ad.document_id);

    auto ev_doc = data.documents.at(event.document_id);
    auto ev_doc_categories = data.document_categories.equal_range(event.document_id);
    //auto ev_doc_topics = data.document_topics.equal_range(event.document_id);
    //auto ev_doc_entities = data.document_entities.equal_range(event.document_id);

    // Start building line
    //line_builder line(label);
    std::vector<ffm_feature> features;


    features.push_back(feature_hashed(0, ad_id));
    features.push_back(feature_hashed(1, ad.campaign_id));
    features.push_back(feature_hashed(2, ad.advertiser_id));

    features.push_back(feature_hashed(3, event.platform));
    features.push_back(feature_hashed(4, event.country));
    features.push_back(feature_hashed(5, event.state));

    // Document info
    features.push_back(feature_hashed(6, event.document_id));
    features.push_back(feature_hashed(7, ev_doc.source_id));
    features.push_back(feature_hashed(8, ev_doc.publisher_id));

    for (auto it = ev_doc_categories.first; it != ev_doc_categories.second; ++ it)
        features.push_back(feature_hashed(12, it->second.first, it->second.second));

    /*
    for (auto it = ev_doc_topics.first; it != ev_doc_topics.second; ++ it)
        line.feature_hashed(14, it->second.first, it->second.second);

    for (auto it = ev_doc_entities.first; it != ev_doc_entities.second; ++ it)
        line.feature_hashed(16, it->second.first, it->second.second);
    */

    // Promoted document info
    features.push_back(feature_hashed(9, ad.document_id));
    features.push_back(feature_hashed(10, ad_doc.source_id));
    features.push_back(feature_hashed(11, ad_doc.publisher_id));

    for (auto it = ad_doc_categories.first; it != ad_doc_categories.second; ++ it)
        features.push_back(feature_hashed(13, it->second.first, it->second.second));

    /*
    for (auto it = ad_doc_topics.first; it != ad_doc_topics.second; ++ it)
        line.feature(15, it->second.first, it->second.second);

    for (auto it = ad_doc_entities.first; it != ad_doc_entities.second; ++ it)
        line.feature(17, it->second.first, it->second.second);
    */

    //

    if (ad_doc.publisher_id == ev_doc.publisher_id)
        features.push_back(feature_raw(18, 0)); // Same publisher

    if (ad_doc.source_id == ev_doc.source_id)
        features.push_back(feature_raw(19, 1)); // Same source

    if (leak_viewed > 0)
        features.push_back(feature_raw(20, 2)); // Viewed

    if (leak_not_viewed > 0)
        features.push_back(feature_raw(21, 3)); // Not viewed

    features.push_back(feature_raw(22, event.weekday + 50));
    features.push_back(feature_raw(23, event.hour + 70));

    features.push_back(feature_raw(24, 4, pos_time_diff(event.timestamp - ad_doc.publish_timestamp)));
    features.push_back(feature_raw(25, 5, time_diff(ev_doc.publish_timestamp - ad_doc.publish_timestamp)));

    // Similarity features
    /*
    for (uint i = 0; i < rows[2].size(); ++ i)
        if (stof(rows[2][i]) > 0)
            features.push_back(feature_raw(26 + i, 6 + i, stof(rows[2][i])));
*/
    // TODO Category, topic and entity intersection
    // TODO Doc timestamp diff

    // Write data
    auto offset = data_out.write(features);

    // Update index
    index.size ++;
    index.labels.push_back(rows[0].size() == 3 ? stof(rows[0][2]) * 2 - 1 : 0);
    index.offsets.push_back(offset);
    index.norms.push_back(norm(features));
    index.groups.push_back(event_id);
}


void writer::finish() {
    ffm_write_index(file_name + ".index", index);
}


int main() {
    using namespace std;

    cout << "Loading reference data..." << endl;
    auto data = load_reference_data();

    cout << "Generating files..." << endl;
    generate_files<reference_data, writer>(data, filesets);

    cout << "Max field: " << max_field << endl;
    cout << "Max index: " << max_index << endl;

    cout << "Done." << endl;
}
