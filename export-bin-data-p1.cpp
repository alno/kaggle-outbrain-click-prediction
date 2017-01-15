#include "util/io.h"
#include "util/data.h"
#include "util/generation.h"
#include "util/helpers.h"

#include "ffm.h"

std::vector<std::pair<std::string, std::string>> files = {
    { "cache/clicks_cv2_train.csv.gz", "cv2_train" },
    { "cache/clicks_cv2_test.csv.gz", "cv2_test" },
    { "cache/clicks_cv1_train.csv.gz", "cv1_train" },
    { "cache/clicks_cv1_test.csv.gz", "cv1_test" },
    { "../input/clicks_train.csv.gz", "full_train" },
    { "../input/clicks_test.csv.gz", "full_test" },
};

std::vector<std::string> features = {
    "leak",
    "viewed_docs", "viewed_categories", "viewed_topics",
    "uid_viewed_ads", "uid_viewed_ad_srcs", "uid_viewed_ad_cats", "uid_viewed_ad_tops",
    "rivals"
};

std::string cur_dataset;

std::unordered_map<int, int> ad_counts;
std::unordered_map<int, int> ad_campaign_counts;
std::unordered_map<int, int> ad_advertiser_counts;
std::unordered_map<int, int> ad_doc_counts;
std::unordered_map<int, int> ad_doc_source_counts;
std::unordered_map<int, int> ad_doc_publisher_counts;
std::unordered_map<int, int> ev_doc_counts;
std::unordered_map<int, int> ev_doc_source_counts;
std::unordered_map<int, int> ev_doc_publisher_counts;
std::unordered_map<int, int> uid_counts;


void load_dataset_data(const std::string & dataset) {
    if (cur_dataset == dataset)
        return;

    cur_dataset = dataset;
    std::cout << "Loading " << dataset << " data..." << std::endl;

    ad_counts = read_map(std::string("cache/ad_counts_") + dataset + std::string(".csv.gz"), read_count);
    ad_campaign_counts = read_map(std::string("cache/ad_campaign_counts_") + dataset + std::string(".csv.gz"), read_count);
    ad_advertiser_counts = read_map(std::string("cache/ad_advertiser_counts_") + dataset + std::string(".csv.gz"), read_count);

    ad_doc_counts = read_map(std::string("cache/ad_doc_counts_") + dataset + std::string(".csv.gz"), read_count);
    ad_doc_source_counts = read_map(std::string("cache/ad_doc_source_counts_") + dataset + std::string(".csv.gz"), read_count);
    ad_doc_publisher_counts = read_map(std::string("cache/ad_doc_publisher_counts_") + dataset + std::string(".csv.gz"), read_count);

    ev_doc_counts = read_map(std::string("cache/ev_doc_counts_") + dataset + std::string(".csv.gz"), read_count);
    ev_doc_source_counts = read_map(std::string("cache/ev_doc_source_counts_") + dataset + std::string(".csv.gz"), read_count);
    ev_doc_publisher_counts = read_map(std::string("cache/ev_doc_publisher_counts_") + dataset + std::string(".csv.gz"), read_count);
    uid_counts = read_map(std::string("cache/uid_counts_") + dataset + std::string(".csv.gz"), read_count);
}


class writer {
    std::string file_name;

    ffm_stream_data_writer data_out;
    ffm_index index;
public:
    writer(const std::string & file_name): file_name(file_name), data_out(file_name + ".data") {
        index.size = 0;
        index.offsets.push_back(0);

        load_dataset_data(file_name.substr(6, file_name.find("_") - 6));
    }

    void write(const reference_data & data, const std::vector<std::vector<std::string>> & rows);
    void finish();
};


void writer::write(const reference_data & data, const std::vector<std::vector<std::string>> & rows) {
    int event_id = stoi(rows[0][0]);
    int ad_id = stoi(rows[0][1]);

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

    // Get counts
    auto ad_count = ad_counts.at(ad_id);
    auto ad_campaign_count = ad_campaign_counts.at(ad.campaign_id);
    auto ad_advertiser_count = ad_advertiser_counts.at(ad.advertiser_id);

    auto ad_doc_count = ad_doc_counts.at(ad.document_id);
    auto ad_doc_source_count = ad_doc_source_counts.at(ad_doc.source_id);
    auto ad_doc_publisher_count = ad_doc_publisher_counts.at(ad_doc.publisher_id);

    auto ev_doc_count = ev_doc_counts.at(event.document_id);
    auto ev_doc_source_count = ev_doc_source_counts.at(ev_doc.source_id);
    auto ev_doc_publisher_count = ev_doc_publisher_counts.at(ev_doc.publisher_id);

    auto uid_count = uid_counts.at(event.uid);

    // Start building line
    ffm_feature_vector_builder features(200);

    // Event features
    features.hashed(0, event.platform);
    features.hashed(1, event.country);
    features.hashed(2, event.state);
    //features.hashed(, event.region);
    features.hashed(3, uid_count < 50 ? uid_count : event.uid + 100);

    // Document info
    features.hashed(4, ev_doc_count < 50 ? ev_doc_count : event.document_id + 100);
    features.hashed(5, ev_doc_source_count < 10 ? ev_doc_source_count : ev_doc.source_id + 10);
    features.hashed(6, ev_doc_publisher_count < 10 ? ev_doc_publisher_count : ev_doc.publisher_id + 10);

    for (auto it = ev_doc_categories.first; it != ev_doc_categories.second; ++ it)
        features.hashed(7, it->second.first, it->second.second);
    /*
    for (auto it = ev_doc_topics.first; it != ev_doc_topics.second; ++ it)
        features.hashed(14, it->second.first, it->second.second);

    for (auto it = ev_doc_entities.first; it != ev_doc_entities.second; ++ it)
        features.hashed(16, it->second.first, it->second.second);
    */

    // Common features

    // Same feature markers

    if (ad_doc.publisher_id == ev_doc.publisher_id)
        features.raw(10, 0); // Same publisher

    if (ad_doc.source_id == ev_doc.source_id)
        features.raw(10, 1); // Same source

    // Document view features (including leak)

    if (stoi(rows[1][0]) > 0)
        features.raw(11, 2); // Viewed ad document (leak)

    if (stoi(rows[1][1]) > 0)
        features.raw(11, 3); // Not viewed ad document (leak)

    if (stoi(rows[2][0]) > 0)
        features.raw(11, 4); // Viewed documents of same publisher

    if (stoi(rows[2][1]) > 0)
        features.raw(11, 5); // Viewed documents of same source

    if (stof(rows[3][0]) > 0)
        features.raw(11, 6); // Viewed documents of the similar category

    if (stof(rows[4][0]) > 0)
        features.raw(11, 7); // Viewed documents of the similar topic

    // Ad view/click features

    auto & v_ad_row = rows[5];
    auto & v_ad_src_row = rows[6];
    auto & v_ad_cat_row = rows[7];
    auto & v_ad_top_row = rows[8];

    if (stoi(v_ad_row[2]) > 0)
        features.raw(12, 20); // Viewed this ad earlier

    if (stoi(v_ad_row[1]) > 0)
        features.raw(12, 21); // Clicked this ad earlier

    if (stoi(v_ad_row[5]) > 0)
        features.raw(12, 22); // Viewed this ad doc earlier

    if (stoi(v_ad_row[4]) > 0)
        features.raw(12, 23); // Clicked this ad doc earlier


    if (stoi(v_ad_src_row[2]) > 0)
        features.raw(12, 24); // Viewed ad of the same publisher earlier

    if (stoi(v_ad_src_row[1]) > 0)
        features.raw(12, 25); // Clicked ad of the same publisher earlier

    if (stoi(v_ad_src_row[5]) > 0)
        features.raw(12, 26); // Viewed ad of the same source earlier

    if (stoi(v_ad_src_row[4]) > 0)
        features.raw(12, 27); // Clicked ad of the same source earlier


    if (stof(v_ad_cat_row[2]) > 0)
        features.raw(12, 28); // Viewed ad of the similar category

    if (stof(v_ad_cat_row[1]) > 0)
        features.raw(12, 29); // Clicked ad of the similar category


    if (stof(v_ad_top_row[2]) > 0)
        features.raw(12, 30); // Viewed ad of the similar topic

    if (stof(v_ad_top_row[1]) > 0)
        features.raw(12, 32); // Clicked ad of the similar topic


    features.raw(13, event.weekday + 50);
    features.raw(13, event.hour + 70);

    features.raw(14, 80, pos_time_diff(event.timestamp - ad_doc.publish_timestamp));
    features.raw(14, 81, time_diff(ev_doc.publish_timestamp - ad_doc.publish_timestamp));

    features.raw(18, stoi(rows[9][0]) + 180); // Rival count

    // Rival ids
    auto rival_ids = split(rows[9][1], ' ');
    for (uint ri = 0; ri < rival_ids.size(); ++ ri) {
        auto rival_id = stoi(rival_ids[ri]);

        if (rival_id != ad_id)
            features.hashed(20, rival_id);
    }

    auto doc_ad_others_it = data.doc_ad_others.find(event_id);
    if (doc_ad_others_it != data.doc_ad_others.end()) {
        auto ids = doc_ad_others_it->second;

        for (uint i = 0; i < ids.size(); ++ i)
            features.hashed(21, ids[i]);
    }

    // Similarity features
    /*
    for (uint i = 0; i < rows[2].size(); ++ i)
        if (stof(rows[2][i]) > 0)
            features.raw(26 + i, 6 + i, stof(rows[2][i]));
    */

    // Ad features
    features.hashed(30, ad_count < 50 ? ad_count : ad_id + 100);
    features.hashed(31, ad_campaign_count < 50 ? ad_campaign_count : ad.campaign_id + 100);
    features.hashed(32, ad_advertiser_count < 50 ? ad_advertiser_count : ad.advertiser_id + 100);

    // Promoted document info
    features.hashed(33, ad_doc_count < 50 ? ad_doc_count : ad.document_id + 100);
    features.hashed(34, ad_doc_source_count < 10 ? ad_doc_source_count : ad_doc.source_id + 10);
    features.hashed(35, ad_doc_publisher_count < 10 ? ad_doc_publisher_count : ad_doc.publisher_id + 10);

    for (auto it = ad_doc_categories.first; it != ad_doc_categories.second; ++ it)
        features.hashed(36, it->second.first, it->second.second);
    /*
    for (auto it = ad_doc_topics.first; it != ad_doc_topics.second; ++ it)
        features.hashed(15, it->second.first, it->second.second);

    for (auto it = ad_doc_entities.first; it != ad_doc_entities.second; ++ it)
        features.hashed(17, it->second.first, it->second.second);
    */

    // Write data
    auto offset = data_out.write(features.data());

    // Update index
    index.size ++;
    index.labels.push_back(rows[0].size() == 3 ? stof(rows[0][2]) * 2 - 1 : 0);
    index.offsets.push_back(offset);
    index.norms.push_back(features.norm());
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
    generate_files<reference_data, writer>(data, build_filesets(files, features, "_bin_p1"));

    cout << "Done." << endl;
}
