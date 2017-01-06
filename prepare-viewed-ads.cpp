#include "util/io.h"
#include "util/data.h"


std::vector<std::pair<std::string, std::string>> filesets {
    std::make_pair("cache/clicks_val_train.csv.gz", "val_train"),
    std::make_pair("cache/clicks_val_test.csv.gz", "val_test"),
    std::make_pair("../input/clicks_train.csv.gz", "full_train"),
    std::make_pair("../input/clicks_test.csv.gz", "full_test"),
};


std::streamsize buffer_size = 1024*1024;

std::pair<int, int> read_event_uid(const std::vector<std::string> & row) {
    return std::make_pair(stoi(row[0]), stoi(row[11]));
}

std::pair<int, int> read_ad_document(const std::vector<std::string> & row) {
    return std::make_pair(stoi(row[0]), stoi(row[1]));
}


std::vector<int> event_uids;
std::vector<ad> ads;
std::unordered_map<int, document> documents;
std::unordered_multimap<int, std::pair<int, float>> document_categories;
std::unordered_multimap<int, std::pair<int, float>> document_topics;


class basic_writer {
    std::unordered_map<std::pair<int, int>, std::pair<uint8_t, uint8_t>> ad_counts;
    std::unordered_map<std::pair<int, int>, std::pair<uint8_t, uint8_t>> ad_doc_counts;
public:

    std::string get_header() {
        return "ad_view_count,ad_click_count,ad_doc_view_count,ad_doc_click_count";
    }

    void write(std::ostream & out, int event_id, int ad_id) {
        using namespace std;

        auto uid = event_uids[event_id];

        auto doc_id = ads[ad_id].document_id;

        auto & ad_cnt = ad_counts[make_pair(uid, ad_id)];
        auto & ad_doc_cnt = ad_doc_counts[make_pair(uid, doc_id)];

        out << int(ad_cnt.first) << ","
            << int(ad_cnt.second) << ","
            << int(ad_doc_cnt.first) << ","
            << int(ad_doc_cnt.second) << endl;
    }


    void update(int event_id, int ad_id, int clicked) {
        using namespace std;

        auto uid = event_uids[event_id];

        auto doc_id = ads[ad_id].document_id;

        auto & ad_cnt = ad_counts[make_pair(uid, ad_id)];
        auto & ad_doc_cnt = ad_doc_counts[make_pair(uid, doc_id)];

        if (int(ad_cnt.first) > 250 || int(ad_doc_cnt.first) > 250)
            throw std::logic_error("Overflow is near");

        ++ ad_cnt.first;
        ++ ad_doc_cnt.first;

        if (clicked > 0) {
            ++ ad_cnt.second;
            ++ ad_doc_cnt.second;
        }
    }

};


class source_writer {
    std::unordered_map<std::pair<int, int>, std::pair<uint8_t, uint8_t>> ad_pub_counts;
    std::unordered_map<std::pair<int, int>, std::pair<uint8_t, uint8_t>> ad_src_counts;
public:

    std::string get_header() {
        return "ad_publisher_view_count,ad_publisher_click_count,ad_source_view_count,ad_source_click_count";
    }

    void write(std::ostream & out, int event_id, int ad_id) {
        using namespace std;

        auto uid = event_uids[event_id];

        auto doc_id = ads[ad_id].document_id;
        auto doc = documents.at(doc_id);

        auto & ad_pub_cnt = ad_pub_counts[make_pair(uid, doc.publisher_id)];
        auto & ad_src_cnt = ad_src_counts[make_pair(uid, doc.source_id)];

        out << int(ad_pub_cnt.first) << ","
            << int(ad_pub_cnt.second) << ","
            << int(ad_src_cnt.first) << ","
            << int(ad_src_cnt.second) << endl;
    }

    void update(int event_id, int ad_id, int clicked) {
        using namespace std;

        auto uid = event_uids[event_id];

        auto doc_id = ads[ad_id].document_id;
        auto doc = documents.at(doc_id);

        auto & ad_pub_cnt = ad_pub_counts[make_pair(uid, doc.publisher_id)];
        auto & ad_src_cnt = ad_src_counts[make_pair(uid, doc.source_id)];

        if (int(ad_pub_cnt.first) > 250 || int(ad_src_cnt.first) > 250)
            throw std::logic_error("Overflow is near");

        ++ ad_pub_cnt.first;
        ++ ad_src_cnt.first;

        if (clicked > 0) {
            ++ ad_pub_cnt.second;
            ++ ad_src_cnt.second;
        }
    }

};


class campaign_writer {
    std::unordered_map<std::pair<int, int>, std::pair<uint8_t, uint8_t>> ad_campaign_counts;
    std::unordered_map<std::pair<int, int>, std::pair<uint8_t, uint8_t>> ad_advertiser_counts;
public:

    std::string get_header() {
        return "ad_campaign_view_count,ad_campaign_click_count,ad_advertiser_view_count,ad_advertiser_click_count";
    }

    void write(std::ostream & out, int event_id, int ad_id) {
        using namespace std;

        auto uid = event_uids[event_id];
        auto ad = ads[ad_id];

        auto & ad_cmp_cnt = ad_campaign_counts[make_pair(uid, ad.campaign_id)];
        auto & ad_adv_cnt = ad_advertiser_counts[make_pair(uid, ad.advertiser_id)];

        out << int(ad_cmp_cnt.first) << ","
            << int(ad_cmp_cnt.second) << ","
            << int(ad_adv_cnt.first) << ","
            << int(ad_adv_cnt.second) << endl;
    }

    void update(int event_id, int ad_id, int clicked) {
        using namespace std;

        auto uid = event_uids[event_id];
        auto ad = ads[ad_id];

        auto & ad_cmp_cnt = ad_campaign_counts[make_pair(uid, ad.campaign_id)];
        auto & ad_adv_cnt = ad_advertiser_counts[make_pair(uid, ad.advertiser_id)];

        if (int(ad_cmp_cnt.first) > 250 || int(ad_adv_cnt.first) > 250)
            throw std::logic_error("Overflow is near");

        ++ ad_cmp_cnt.first;
        ++ ad_adv_cnt.first;

        if (clicked > 0) {
            ++ ad_cmp_cnt.second;
            ++ ad_adv_cnt.second;
        }
    }

};


class source_ctr_writer {
    std::unordered_map<std::pair<int, int>, std::pair<uint8_t, uint8_t>> ad_pub_counts;
    std::unordered_map<std::pair<int, int>, std::pair<uint8_t, uint8_t>> ad_src_counts;
public:

    std::string get_header() {
        return "ad_publisher_ctr,ad_source_ctr";
    }

    void write(std::ostream & out, int event_id, int ad_id) {
        using namespace std;

        auto uid = event_uids[event_id];

        auto doc_id = ads[ad_id].document_id;
        auto doc = documents.at(doc_id);

        auto & ad_pub_cnt = ad_pub_counts[make_pair(uid, doc.publisher_id)];
        auto & ad_src_cnt = ad_src_counts[make_pair(uid, doc.source_id)];

        const float reg_n = 5;
        const float reg_p = 0.194;

        float ad_pub_ctr = (ad_pub_cnt.second + reg_p * reg_n) / (ad_pub_cnt.first + reg_n);
        float ad_src_ctr = (ad_src_cnt.second + reg_p * reg_n) / (ad_src_cnt.first + reg_n);

        out << ad_pub_ctr << ","
            << ad_src_ctr << endl;
    }

    void update(int event_id, int ad_id, int clicked) {
        using namespace std;

        auto uid = event_uids[event_id];

        auto doc_id = ads[ad_id].document_id;
        auto doc = documents.at(doc_id);

        auto & ad_pub_cnt = ad_pub_counts[make_pair(uid, doc.publisher_id)];
        auto & ad_src_cnt = ad_src_counts[make_pair(uid, doc.source_id)];

        if (int(ad_pub_cnt.first) > 250 || int(ad_src_cnt.first) > 250)
            throw std::logic_error("Overflow is near");

        if (clicked >= 0) {
            ++ ad_pub_cnt.first;
            ++ ad_src_cnt.first;
        }

        if (clicked > 0) {
            ++ ad_pub_cnt.second;
            ++ ad_src_cnt.second;
        }
    }

};



class category_writer {
    std::unordered_map<std::pair<int, int>, std::pair<float, float>> ad_cat_counts;
public:

    std::string get_header() {
        return "ad_category_view_weight,ad_category_click_weight";
    }

    void write(std::ostream & out, int event_id, int ad_id) {
        using namespace std;

        auto uid = event_uids[event_id];
        auto doc_id = ads[ad_id].document_id;
        auto doc_categories = document_categories.equal_range(doc_id);

        float cat_view_weight = 0;
        float cat_click_weight = 0;

        for (auto it = doc_categories.first; it != doc_categories.second; ++ it) {
            auto & ad_cat_cnt = ad_cat_counts[make_pair(uid, it->second.first)];

            cat_view_weight += ad_cat_cnt.first * it->second.second;
            cat_click_weight += ad_cat_cnt.second * it->second.second;
        }

        out << cat_view_weight << ","
            << cat_click_weight << endl;
    }

    void update(int event_id, int ad_id, int clicked) {
        using namespace std;

        auto uid = event_uids[event_id];
        auto doc_id = ads[ad_id].document_id;
        auto doc_categories = document_categories.equal_range(doc_id);

        for (auto it = doc_categories.first; it != doc_categories.second; ++ it) {
            auto & ad_cat_cnt = ad_cat_counts[make_pair(uid, it->second.first)];

            ad_cat_cnt.first += it->second.second;

            if (clicked > 0)
                ad_cat_cnt.second += it->second.second;
        }
    }
};


class topic_writer {
    std::unordered_map<std::pair<int, int>, std::pair<float, float>> ad_top_counts;
public:

    std::string get_header() {
        return "ad_topic_view_weight,ad_topic_click_weight";
    }

    void write(std::ostream & out, int event_id, int ad_id) {
        using namespace std;

        auto uid = event_uids[event_id];
        auto doc_id = ads[ad_id].document_id;
        auto doc_topics = document_topics.equal_range(doc_id);

        float top_view_weight = 0;
        float top_click_weight = 0;

        for (auto it = doc_topics.first; it != doc_topics.second; ++ it) {
            auto & ad_top_cnt = ad_top_counts[make_pair(uid, it->second.first)];

            top_view_weight += ad_top_cnt.first * it->second.second;
            top_click_weight += ad_top_cnt.second * it->second.second;
        }

        out << top_view_weight << ","
            << top_click_weight << endl;
    }

    void update(int event_id, int ad_id, int clicked) {
        using namespace std;

        auto uid = event_uids[event_id];
        auto doc_id = ads[ad_id].document_id;
        auto doc_topics = document_topics.equal_range(doc_id);

        for (auto it = doc_topics.first; it != doc_topics.second; ++ it) {
            auto & ad_top_cnt = ad_top_counts[make_pair(uid, it->second.first)];

            ad_top_cnt.first += it->second.second;

            if (clicked > 0)
                ad_top_cnt.second += it->second.second;
        }
    }
};


struct row {
    int event_id;
    int ad_id;
    int clicked;
};


template <typename W>
void process_group(W & w, const std::vector<row> & group, std::ostream & out) {
    for (auto it = group.begin(); it != group.end(); ++ it)
        w.write(out, it->event_id, it->ad_id);

    for (auto it = group.begin(); it != group.end(); ++ it)
        w.update(it->event_id, it->ad_id, it->clicked);
}


template <typename W>
void generate(const std::string & a_in_file_name, const std::string & b_in_file_name, const std::string & a_out_file_name, const std::string & b_out_file_name) {
    using namespace std;

    cout << "Generating " << a_out_file_name<< " and " << b_out_file_name << "... ";
    cout.flush();

    clock_t begin = clock();

    W w;

    compressed_csv_file a_in(a_in_file_name);
    compressed_csv_file b_in(b_in_file_name);

    boost::iostreams::filtering_ostream a_out;
    a_out.push(boost::iostreams::gzip_compressor(), buffer_size, buffer_size);
    a_out.push(boost::iostreams::file_sink(a_out_file_name, std::ios_base::out | std::ios_base::binary), buffer_size, buffer_size);
    a_out << w.get_header() << endl;

    boost::iostreams::filtering_ostream b_out;
    b_out.push(boost::iostreams::gzip_compressor(), buffer_size, buffer_size);
    b_out.push(boost::iostreams::file_sink(b_out_file_name, std::ios_base::out | std::ios_base::binary), buffer_size, buffer_size);
    b_out << w.get_header() << endl;

    auto a_row = a_in.getrow();
    auto b_row = b_in.getrow();

    uint i = 0;

    vector<row> group;
    int group_id = -1;
    boost::iostreams::filtering_ostream * group_out = nullptr;

    while (!a_row.empty() || !b_row.empty()) {
        if (b_row.empty() || (!a_row.empty() && stoi(a_row[0]) < stoi(b_row[0]))) {
            row r;

            r.event_id = stoi(a_row[0]);
            r.ad_id = stoi(a_row[1]);
            r.clicked = stoi(a_row[2]);

            if (r.event_id != group_id) {
                process_group(w, group, *group_out);

                group.clear();
                group_id = r.event_id;
                group_out = &a_out;
            }

            group.push_back(r);

            a_row = a_in.getrow();
        } else {
            row r;

            r.event_id = stoi(b_row[0]);
            r.ad_id = stoi(b_row[1]);
            r.clicked = -1;

            if (r.event_id != group_id) {
                process_group(w, group, *group_out);

                group.clear();
                group_id = r.event_id;
                group_out = &b_out;
            }

            group.push_back(r);

            b_row = b_in.getrow();
        }

        ++ i;

        if (i % 5000000 == 0) {
            cout << (i / 1000000) << "M... ";
            cout.flush();
        }
    }

    process_group(w, group, *group_out);

    clock_t end = clock();
    double elapsed = double(end - begin) / CLOCKS_PER_SEC;

    cout << "done in " << elapsed << " seconds" << endl;
}


int main() {
    using namespace std;

    cout << "Loading reference data..." << endl;
    event_uids = read_vector("cache/events.csv.gz", read_event_uid, 23120127);
    ads = read_ads();
    documents = read_map("cache/documents.csv.gz", read_document);
    document_categories = read_multi_map("../input/documents_categories.csv.gz", read_document_annotation);
    document_topics = read_multi_map("../input/documents_topics.csv.gz", read_document_annotation);

    for (uint ofs = 0; ofs < filesets.size(); ofs += 2) {
        generate<basic_writer>(
            filesets[ofs].first,
            filesets[ofs+1].first,
            string("cache/viewed_ads_") + filesets[ofs].second + string(".csv.gz"),
            string("cache/viewed_ads_") + filesets[ofs+1].second + string(".csv.gz")
        );

        generate<source_writer>(
            filesets[ofs].first,
            filesets[ofs+1].first,
            string("cache/viewed_ad_srcs_") + filesets[ofs].second + string(".csv.gz"),
            string("cache/viewed_ad_srcs_") + filesets[ofs+1].second + string(".csv.gz")
        );

        generate<campaign_writer>(
            filesets[ofs].first,
            filesets[ofs+1].first,
            string("cache/viewed_ad_campaigns_") + filesets[ofs].second + string(".csv.gz"),
            string("cache/viewed_ad_campaigns_") + filesets[ofs+1].second + string(".csv.gz")
        );

        generate<source_ctr_writer>(
            filesets[ofs].first,
            filesets[ofs+1].first,
            string("cache/viewed_ad_src_ctrs_") + filesets[ofs].second + string(".csv.gz"),
            string("cache/viewed_ad_src_ctrs_") + filesets[ofs+1].second + string(".csv.gz")
        );

        generate<category_writer>(
            filesets[ofs].first,
            filesets[ofs+1].first,
            string("cache/viewed_ad_categories_") + filesets[ofs].second + string(".csv.gz"),
            string("cache/viewed_ad_categories_") + filesets[ofs+1].second + string(".csv.gz")
        );

        generate<topic_writer>(
            filesets[ofs].first,
            filesets[ofs+1].first,
            string("cache/viewed_ad_topics_") + filesets[ofs].second + string(".csv.gz"),
            string("cache/viewed_ad_topics_") + filesets[ofs+1].second + string(".csv.gz")
        );
    }

    cout << "Done." << endl;
}
