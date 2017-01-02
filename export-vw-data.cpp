#include "util/io.h"
#include "util/data.h"
#include "util/generation.h"

std::vector<std::pair<std::vector<std::string>, std::string>> filesets = {
    { { "cache/clicks_val_train.csv.gz", "cache/leak_val_train.csv.gz", "cache/similarity_val_train.csv.gz" }, "cache/val_train_vw.txt" },
    { { "cache/clicks_val_test.csv.gz", "cache/leak_val_test.csv.gz", "cache/similarity_val_test.csv.gz" }, "cache/val_test_vw.txt" },
    { { "../input/clicks_train.csv.gz", "cache/leak_full_train.csv.gz", "cache/similarity_full_train.csv.gz" }, "cache/full_train_vw.txt" },
    { { "../input/clicks_test.csv.gz", "cache/leak_full_test.csv.gz", "cache/similarity_full_test.csv.gz" }, "cache/full_test_vw.txt" },
};

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
    auto ad_doc_topics = data.document_topics.equal_range(ad.document_id);
    auto ad_doc_entities = data.document_entities.equal_range(ad.document_id);

    auto ev_doc = data.documents.at(event.document_id);
    auto ev_doc_categories = data.document_categories.equal_range(event.document_id);
    auto ev_doc_topics = data.document_topics.equal_range(event.document_id);
    auto ev_doc_entities = data.document_entities.equal_range(event.document_id);

    std::stringstream line;

    if (label >= 0)
        line << (label * 2 - 1) << " ";

    line << "|a ad_" << ad_id << " ac_" << ad.campaign_id << " aa_" << ad.advertiser_id;
    line << "|l c_" << event.country << " s_" << event.state << " p_" << event.platform;
    line << "|t h_" << event.hour << " w_" << event.weekday;
    line << "|u u_" << event.uid;

    // Document info
    line << "|d ed_" << event.document_id << " eds_" << ev_doc.source_id << " edp_" << ev_doc.publisher_id;

    for (auto it = ev_doc_categories.first; it != ev_doc_categories.second; ++ it)
        line << " edc_" << it->second.first << ":" << it->second.second;

    for (auto it = ev_doc_topics.first; it != ev_doc_topics.second; ++ it)
        line << " edt_" << it->second.first << ":" << it->second.second;

    for (auto it = ev_doc_entities.first; it != ev_doc_entities.second; ++ it)
        line << " ede_" << it->second.first << ":" << it->second.second;

    // Promoted document info
    line << "|p ad_" << ad.document_id << " ads_" << ad_doc.source_id << " adp_" << ad_doc.publisher_id;

    for (auto it = ad_doc_categories.first; it != ad_doc_categories.second; ++ it)
        line << " adc_" << it->second.first << ":" << it->second.second;

    for (auto it = ad_doc_topics.first; it != ad_doc_topics.second; ++ it)
        line << " adt_" << it->second.first << ":" << it->second.second;

    for (auto it = ad_doc_entities.first; it != ad_doc_entities.second; ++ it)
        line << " ade_" << it->second.first << ":" << it->second.second;

    // Manual features
    line << "|f";

    if (ad_doc.publisher_id == ev_doc.publisher_id)
        line << " sp"; // Same publisher

    if (ad_doc.source_id == ev_doc.source_id)
        line << " ss"; // Same source

    // Leak features
    if (leak_viewed > 0)
        line << " v"; // Same source

    if (leak_not_viewed > 0)
        line << " nv"; // Same source

    // Similarity features
    for (int i = 0; i < rows[2].size(); ++ i)
        line << " s_" << i << ':' << rows[2][i];

    line << std::endl;

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
