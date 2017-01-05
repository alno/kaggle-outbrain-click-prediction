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
    return std::make_pair(stoi(row[0]), stoi(row[10]));
}

std::pair<int, int> read_ad_document(const std::vector<std::string> & row) {
    return std::make_pair(stoi(row[0]), stoi(row[1]));
}


std::vector<int> event_uids;
std::vector<int> ad_doc_ids;
std::unordered_map<int, document> documents;


class basic_writer {
    std::unordered_map<std::pair<int, int>, std::pair<uint8_t, uint8_t>> ad_counts;
    std::unordered_map<std::pair<int, int>, std::pair<uint8_t, uint8_t>> ad_doc_counts;
public:

    std::string get_header() {
        return "ad_view_count,ad_click_count,ad_doc_view_count,ad_doc_click_count";
    }

    void process_row(std::ostream & out, int event_id, int ad_id, bool clicked) {
        using namespace std;

        auto uid = event_uids[event_id];

        auto doc_id = ad_doc_ids.at(ad_id);

        auto & ad_cnt = ad_counts[make_pair(uid, ad_id)];
        auto & ad_doc_cnt = ad_doc_counts[make_pair(uid, doc_id)];

        out << int(ad_cnt.first) << ","
            << int(ad_cnt.second) << ","
            << int(ad_doc_cnt.first) << ","
            << int(ad_doc_cnt.second) << endl;

        if (int(ad_cnt.first) > 250 || int(ad_doc_cnt.first) > 250)
            throw std::logic_error("Overflow is near");

        ++ ad_cnt.first;
        ++ ad_doc_cnt.first;

        if (clicked) {
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

    void process_row(std::ostream & out, int event_id, int ad_id, bool clicked) {
        using namespace std;

        auto uid = event_uids[event_id];

        auto doc_id = ad_doc_ids.at(ad_id);
        auto doc = documents.at(doc_id);

        auto & ad_pub_cnt = ad_pub_counts[make_pair(uid, doc.publisher_id)];
        auto & ad_src_cnt = ad_src_counts[make_pair(uid, doc.source_id)];

        out << int(ad_pub_cnt.first) << ","
            << int(ad_pub_cnt.second) << ","
            << int(ad_src_cnt.first) << ","
            << int(ad_src_cnt.second) << endl;

        if (int(ad_pub_cnt.first) > 250 || int(ad_src_cnt.first) > 250)
            throw std::logic_error("Overflow is near");

        ++ ad_pub_cnt.first;
        ++ ad_src_cnt.first;

        if (clicked) {
            ++ ad_pub_cnt.second;
            ++ ad_src_cnt.second;
        }
    }

};


template <typename W>
void generate(const std::string & a_in_file_name, const std::string & b_in_file_name, const std::string & a_out_file_name, const std::string & b_out_file_name) {
    using namespace std;

    cout << "Generating " << a_out_file_name<< " and " << b_out_file_name << "... ";
    cout.flush();

    clock_t begin = clock();

    W w;

    unordered_map<pair<int, int>, pair<uint8_t, uint8_t>> ad_counts;
    unordered_map<pair<int, int>, pair<uint8_t, uint8_t>> ad_doc_counts;
    unordered_map<pair<int, int>, pair<uint8_t, uint8_t>> ad_pub_counts;
    unordered_map<pair<int, int>, pair<uint8_t, uint8_t>> ad_src_counts;

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

    while (!a_row.empty() || !b_row.empty()) {
        if (b_row.empty() || (!a_row.empty() && stoi(a_row[0]) < stoi(b_row[0]))) {
            w.process_row(a_out, stoi(a_row[0]), stoi(a_row[1]), stoi(a_row[2]) > 0);
            a_row = a_in.getrow();
        } else {
            w.process_row(b_out, stoi(b_row[0]), stoi(b_row[1]), false);
            b_row = b_in.getrow();
        }

        ++ i;

        if (i % 5000000 == 0) {
            cout << (i / 1000000) << "M... ";
            cout.flush();
        }
    }

    clock_t end = clock();
    double elapsed = double(end - begin) / CLOCKS_PER_SEC;

    cout << "done in " << elapsed << " seconds" << endl;
}


int main() {
    using namespace std;

    cout << "Loading reference data..." << endl;
    event_uids = read_vector("cache/events.csv.gz", read_event_uid, 23120127);
    ad_doc_ids = read_vector("../input/promoted_content.csv.gz", read_ad_document, 573099);
    documents = read_map("cache/documents.csv.gz", read_document);

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
    }

    cout << "Done." << endl;
}
