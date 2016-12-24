#include "util/io.h"
#include "util/data.h"
#include "util/generation.h"

#include <functional>

std::vector<std::pair<std::string, std::string>> filesets = {
    std::make_pair("cache/clicks_val_train.csv.gz", "cache/val_train_ffm.txt"),
    std::make_pair("cache/clicks_val_test.csv.gz", "cache/val_test_ffm.txt"),
    std::make_pair("../input/clicks_train.csv.gz", "cache/full_train_ffm.txt"),
    std::make_pair("../input/clicks_test.csv.gz", "cache/full_test_ffm.txt"),
};

struct reference_data {
    std::vector<event> events;
    std::vector<ad> ads;
    std::unordered_map<int, document> documents;
};

reference_data load_reference_data() {
    reference_data res;
    res.events = read_vector("cache/events.csv.gz", read_event, 23120127);
    res.ads = read_vector("../input/promoted_content.csv.gz", read_ad, 573099);
    res.documents = read_map("cache/documents.csv.gz", read_document);

    return res;
}

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

std::string encode_row(const reference_data & data, int event_id, int ad_id, int label) {
    auto ad = data.ads[ad_id];
    auto event = data.events[event_id];

    std::stringstream line;

    line << label << " ";
    line << "0:" << h(ad_id) << ":1 1:" << event.platform << ":1 2:" << h(ad.campaign_id) << ":1 3:" << h(ad.advertiser_id) << ":1 ";
    line << "4:" << h(event.country) << ":1 5:" << h(event.state) << ":1 ";
    line << "6:" << h(ad.document_id) << ":1 7:" << h(event.document_id) << ":1 ";
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
