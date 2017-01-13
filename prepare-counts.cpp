#include "util/io.h"
#include "util/data.h"

std::vector<std::pair<std::string, std::vector<std::string>>> filesets {
    { "cv1", { "cache/clicks_cv1_train.csv.gz", "cache/clicks_cv1_test.csv.gz" } },
    { "cv2", { "cache/clicks_cv2_train.csv.gz", "cache/clicks_cv2_test.csv.gz" } },
    { "full", { "../input/clicks_train.csv.gz", "../input/clicks_test.csv.gz" } }
};


struct cnt {
    uint32_t train_count;
    uint32_t test_count;

    uint32_t & operator[](uint i) {
        switch (i) {
            case 0: return train_count;
            case 1: return test_count;
            default: throw std::logic_error("Invalid field index");
        }
    }
};


template <typename T>
void write_counts(const std::unordered_map<T, cnt> & map, const std::string & file_name) {
    using namespace std;

    cout << "Writing " << file_name << "... " << endl;

    ofstream outfile(file_name, std::ios_base::out | std::ios_base::binary);

    streamsize buffer_size = 1024*1024;
    boost::iostreams::filtering_streambuf<boost::iostreams::output> buf;
    buf.push(boost::iostreams::gzip_compressor(), buffer_size, buffer_size);
    buf.push(outfile, buffer_size, buffer_size);

    std::ostream out(&buf);

    out << "id,train_count,test_count" << endl;

    for (auto it = map.begin(); it != map.end(); ++ it)
        out << it->first << "," << it->second.train_count << "," << it->second.test_count << endl;
}


int main() {
    using namespace std;

    cout << "Loading reference data..." << endl;
    auto data = load_reference_data();

    for (auto it = filesets.begin(); it != filesets.end(); ++ it) {
        unordered_map<int, cnt> ad_counts;
        unordered_map<int, cnt> ad_campaign_counts;
        unordered_map<int, cnt> ad_advertiser_counts;
        unordered_map<int, cnt> ad_doc_counts;
        unordered_map<int, cnt> ad_doc_source_counts;
        unordered_map<int, cnt> ad_doc_publisher_counts;
        unordered_map<int, cnt> ev_doc_counts;
        unordered_map<int, cnt> ev_doc_source_counts;
        unordered_map<int, cnt> ev_doc_publisher_counts;
        unordered_map<int, cnt> uid_counts;

        unordered_map<std::string, cnt> ev_country_counts;
        unordered_map<std::string, cnt> ev_state_counts;
        unordered_map<std::string, cnt> ev_region_counts;


        cout << "Processing " << it->first << "... " << endl;

        for (uint fi = 0; fi < it->second.size(); ++ fi) {
            auto file_name = it->second[fi];
            clock_t begin = clock();

            cout << "  Loading " << file_name << "... ";
            cout.flush();

            compressed_csv_file file(file_name);

            for (int ri = 0;; ++ri) {
                auto row = file.getrow();

                if (row.empty())
                    break;

                if (ri > 0 && ri % 5000000 == 0) {
                    cout << (ri / 1000000) << "M... ";
                    cout.flush();
                }

                // Extract fields
                int ev_id = stoi(row[0]);
                int ad_id = stoi(row[1]);

                auto ad = data.ads[ad_id];
                auto ev = data.events[ev_id];

                auto ad_doc = data.documents.at(ad.document_id);
                auto ev_doc = data.documents.at(ev.document_id);

                // Increment counters
                ++ ad_counts[ad_id][fi];
                ++ ad_campaign_counts[ad.campaign_id][fi];
                ++ ad_advertiser_counts[ad.advertiser_id][fi];

                ++ ad_doc_counts[ad.document_id][fi];
                ++ ad_doc_source_counts[ad_doc.source_id][fi];
                ++ ad_doc_publisher_counts[ad_doc.publisher_id][fi];

                ++ ev_doc_counts[ev.document_id][fi];
                ++ ev_doc_source_counts[ev_doc.source_id][fi];
                ++ ev_doc_publisher_counts[ev_doc.publisher_id][fi];

                ++ ev_country_counts[ev.country][fi];
                ++ ev_state_counts[ev.state][fi];
                ++ ev_region_counts[ev.region][fi];

                ++ uid_counts[ev.uid][fi];
            }

            clock_t end = clock();
            double elapsed = double(end - begin) / CLOCKS_PER_SEC;

            cout << "done in " << elapsed << " seconds" << endl;
        }

        write_counts(ad_counts, string("cache/counts/ads_") + it->first + string(".csv.gz"));
        write_counts(ad_campaign_counts, string("cache/counts/ad_campaigns_") + it->first + string(".csv.gz"));
        write_counts(ad_advertiser_counts, string("cache/counts/ad_advertisers_") + it->first + string(".csv.gz"));

        write_counts(ad_doc_counts, string("cache/counts/ad_docs_") + it->first + string(".csv.gz"));
        write_counts(ad_doc_source_counts, string("cache/counts/ad_doc_sources_") + it->first + string(".csv.gz"));
        write_counts(ad_doc_publisher_counts, string("cache/counts/ad_doc_publishers_") + it->first + string(".csv.gz"));

        write_counts(ev_doc_counts, string("cache/counts/ev_docs_") + it->first + string(".csv.gz"));
        write_counts(ev_doc_source_counts, string("cache/counts/ev_doc_sources_") + it->first + string(".csv.gz"));
        write_counts(ev_doc_publisher_counts, string("cache/counts/ev_doc_publishers_") + it->first + string(".csv.gz"));

        write_counts(ev_country_counts, string("cache/counts/ev_countries_") + it->first + string(".csv.gz"));
        write_counts(ev_state_counts, string("cache/counts/ev_states_") + it->first + string(".csv.gz"));
        write_counts(ev_region_counts, string("cache/counts/ev_regions_") + it->first + string(".csv.gz"));

        write_counts(uid_counts, string("cache/counts/uids_") + it->first + string(".csv.gz"));
    }

    cout << "Done." << endl;
}
