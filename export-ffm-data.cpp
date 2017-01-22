#include "util/io.h"
#include "util/data.h"
#include "util/generation.h"

#include <functional>
#include <cmath>

std::vector<std::pair<std::vector<std::string>, std::string>> filesets = {
    { { "cache/clicks_cv2_train.csv.gz", "cache/leak_cv2_train.csv.gz" }, "cache/cv2_train_ffm.txt" },
    { { "cache/clicks_cv2_test.csv.gz", "cache/leak_cv2_test.csv.gz" }, "cache/cv2_test_ffm.txt" },
    { { "cache/clicks_cv1_train.csv.gz", "cache/leak_val_train.csv.gz" }, "cache/cv1_train_ffm.txt" },
    { { "cache/clicks_cv1_test.csv.gz", "cache/leak_val_test.csv.gz" }, "cache/cv1_test_ffm.txt" },
    { { "../input/clicks_train.csv.gz", "cache/leak_full_train.csv.gz" }, "cache/full_train_ffm.txt" },
    { { "../input/clicks_test.csv.gz", "cache/leak_full_test.csv.gz" }, "cache/full_test_ffm.txt" },
};

std::hash<std::string> str_hash;

uint32_t hash_offset = 100;
uint32_t hash_base = 1 << 19;

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

class line_builder {
public:
    std::stringstream stream;

    line_builder(int label) {
        stream << label;
    }

    void feature(uint32_t field, uint32_t category) {
        stream << ' ' << field << ':' << h(category, field) << ":1";
    }

    void feature(uint32_t field, uint32_t category, float value) {
        stream << ' ' << field << ':' << h(category, field) << ':' << value;
    }

    void feature(uint32_t field, const std::string & category) {
        stream << ' ' << field << ':' << h(category, field) << ":1";
    }

    void feature(uint32_t field, const std::string & category, float value) {
        stream << ' ' << field << ':' << h(category, field) << ':' << value;
    }

    void append(const char * str) {
        stream << str;
    }

    std::string str() {
        return stream.str();
    }
};


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


class writer {
    std::ofstream out;
public:
    writer(const std::string & file_name) : out(file_name) {}

    void write(const reference_data & data, const std::vector<std::vector<std::string>> & rows);
    void finish() {}
};


void writer::write(const reference_data & data, const std::vector<std::vector<std::string>> & rows) {
    int event_id = stoi(rows[0][0]);
    int ad_id = stoi(rows[0][1]);
    int label = rows[0].size() == 3 ? stoi(rows[0][2]) : -1;

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
    line_builder line(label);

    line.feature(0, ad_id);
    line.feature(1, ad.campaign_id);
    line.feature(2, ad.advertiser_id);

    line.feature(3, event.platform);
    line.feature(4, event.country);
    line.feature(5, event.state);

    // Document info
    line.feature(6, event.document_id);
    line.feature(7, ev_doc.source_id);
    line.feature(8, ev_doc.publisher_id);

    for (auto it = ev_doc_categories.first; it != ev_doc_categories.second; ++ it)
        line.feature(12, it->second.first, it->second.second);

    // Promoted document info
    line.feature(9, ad.document_id);
    line.feature(10, ad_doc.source_id);
    line.feature(11, ad_doc.publisher_id);

    for (auto it = ad_doc_categories.first; it != ad_doc_categories.second; ++ it)
        line.feature(13, it->second.first, it->second.second);

    // Manual features

    if (ad_doc.publisher_id == ev_doc.publisher_id)
        line.append(" 18:0:1"); // Same publisher

    if (ad_doc.source_id == ev_doc.source_id)
        line.append(" 19:1:1"); // Same source

    if (leak_viewed > 0)
        line.append(" 20:2:1"); // Same source

    if (leak_not_viewed > 0)
        line.append(" 21:3:1"); // Same source

    line.stream << " 22:" << (event.weekday + 50) << ":1 23:" << (event.hour + 70) << ":1";

    line.stream << " 24:4:" << pos_time_diff(event.timestamp - ad_doc.publish_timestamp);
    line.stream << " 25:5:" << time_diff(ev_doc.publish_timestamp - ad_doc.publish_timestamp);
    line.append("\n");

    out << line.str();
}

int main() {
    using namespace std;

    cout << "Loading reference data..." << endl;
    auto data = load_reference_data();

    cout << "Generating files..." << endl;
    generate_files<reference_data, writer>(data, filesets);

    cout << "Done." << endl;
}
