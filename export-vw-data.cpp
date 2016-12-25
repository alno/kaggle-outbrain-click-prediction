#include "util/io.h"
#include "util/data.h"
#include "util/generation.h"

std::vector<std::pair<std::string, std::string>> filesets = {
    std::make_pair("cache/clicks_val_train.csv.gz", "cache/val_train_vw.txt"),
    std::make_pair("cache/clicks_val_test.csv.gz", "cache/val_test_vw.txt"),
    std::make_pair("../input/clicks_train.csv.gz", "cache/full_train_vw.txt"),
    std::make_pair("../input/clicks_test.csv.gz", "cache/full_test_vw.txt"),
};

struct reference_data {
    std::vector<event> events;
    std::vector<ad> ads;
    std::unordered_map<int, document> documents;
    std::unordered_multimap<int, std::pair<int, float>> document_categories;
};

reference_data load_reference_data() {
    reference_data res;
    res.events = read_vector("cache/events.csv.gz", read_event, 23120127);
    res.ads = read_vector("../input/promoted_content.csv.gz", read_ad, 573099);
    res.documents = read_map("cache/documents.csv.gz", read_document);
    res.document_categories = read_multi_map("../input/document_categories.csv.gz", read_document_category);

    return res;
}

std::string encode_row(const reference_data & data, int event_id, int ad_id, int label) {
    auto ad = data.ads[ad_id];
    auto event = data.events[event_id];

    auto ad_doc = data.documents.at(ad.document_id);
    auto ad_doc_categories = data.document_categories.equal_range(ad.document_id);

    auto ev_doc = data.documents.at(event.document_id);
    auto ev_doc_categories = data.document_categories.equal_range(event.document_id);

    std::stringstream line;

    if (label >= 0)
        line << (label * 2 - 1) << " ";

    line << "|a ad_" << ad_id << " p_" << event.platform << " ac_" << ad.campaign_id << " aa_" << ad.advertiser_id;
    line << "|l c_" << event.country << " s_" << event.state;

    // Document info
    line << "|d ed_" << event.document_id << " eds_" << ev_doc.source_id << " edp_" << ev_doc.publisher_id;

    for (auto it = ev_doc_categories.first; it != ev_doc_categories.second; ++ it)
        line << " edc_" << it->second.first << ":" << it->second.second;

    // Promoted document info
    line << "|p ad_" << ad.document_id << " ads_" << ad_doc.source_id << " adp_" << ad_doc.publisher_id;

    for (auto it = ad_doc_categories.first; it != ad_doc_categories.second; ++ it)
        line << " adc_" << it->second.first << ":" << it->second.second;

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
