#include "util/io.h"
#include "util/data.h"
#include "util/generation.h"

#include <functional>

std::vector<std::pair<std::vector<std::string>, std::string>> filesets = {
    { { "cache/clicks_val_train.csv.gz", "cache/leak_val_train.csv.gz" }, "cache/val_train_ffm.txt" },
    { { "cache/clicks_val_test.csv.gz", "cache/leak_val_test.csv.gz" }, "cache/val_test_ffm.txt" },
    { { "../input/clicks_train.csv.gz", "cache/leak_full_train.csv.gz" }, "cache/full_train_ffm.txt" },
    { { "../input/clicks_test.csv.gz", "cache/leak_full_test.csv.gz" }, "cache/full_test_ffm.txt" },
};

std::hash<std::string> str_hash;

uint32_t hash_base = 1 << 18;

uint32_t h(uint32_t a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a % hash_base;
}

uint32_t h(const std::string & a) {
    return str_hash(a) % hash_base;
}

std::string encode_row(const reference_data & data, const std::vector<std::vector<std::string>> & rows) {
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
    auto ad_doc_topics = data.document_topics.equal_range(ad.document_id);
    auto ad_doc_entities = data.document_entities.equal_range(ad.document_id);

    auto ev_doc = data.documents.at(event.document_id);
    auto ev_doc_categories = data.document_categories.equal_range(event.document_id);
    auto ev_doc_topics = data.document_topics.equal_range(event.document_id);
    auto ev_doc_entities = data.document_entities.equal_range(event.document_id);

    std::stringstream line;

    line << label;
    line << " 0:" << h(ad_id) << ":1 1:" << event.platform << ":1 2:" << h(ad.campaign_id) << ":1 3:" << h(ad.advertiser_id) << ":1";
    line << " 4:" << h(event.country) << ":1 5:" << h(event.state) << ":1 ";

    // Document info
    line << " 6:" << h(event.document_id) << ":1 7:" << h(ev_doc.source_id) << ":1 8:" << h(ev_doc.publisher_id) << ":1";

    for (auto it = ev_doc_categories.first; it != ev_doc_categories.second; ++ it)
        line << " 12:" << h(it->second.first) << ":" << it->second.second;

    for (auto it = ev_doc_topics.first; it != ev_doc_topics.second; ++ it)
        line << " 14:" << h(it->second.first) << ":" << it->second.second;

    for (auto it = ev_doc_entities.first; it != ev_doc_entities.second; ++ it)
        line << " 16:" << h(it->second.first) << ":" << it->second.second;

    // Promoted document info
    line << " 9:" << h(ad.document_id) << ":1 10:" << h(ad_doc.source_id) << ":1 11:" << h(ad_doc.publisher_id) << ":1";

    for (auto it = ad_doc_categories.first; it != ad_doc_categories.second; ++ it)
        line << " 13:" << h(it->second.first) << ":" << it->second.second;

    for (auto it = ad_doc_topics.first; it != ad_doc_topics.second; ++ it)
        line << " 15:" << h(it->second.first) << ":" << it->second.second;

    for (auto it = ad_doc_entities.first; it != ad_doc_entities.second; ++ it)
        line << " 17:" << h(it->second.first) << ":" << it->second.second;

    if (ad_doc.publisher_id == ev_doc.publisher_id)
        line << " 18:1001:1"; // Same publisher

    if (ad_doc.source_id == ev_doc.source_id)
        line << " 19:1002:1"; // Same source

    if (leak_viewed > 0)
        line << " 20:1003:1"; // Same source

    if (leak_not_viewed > 0)
        line << " 21:1004:1"; // Same source

    // TODO Category, topic and entity intersection
    // TODO Weekday, hour
    // TODO Doc timestamp diff

    line << std::endl;

    return line.str();
}

int main() {
    using namespace std;

    cout << "Loading reference data..." << endl;
    auto data = load_reference_data();

    cout << "Generating files..." << endl;
    generate_files(data, filesets, encode_row);

    cout << "Done." << endl;
}
