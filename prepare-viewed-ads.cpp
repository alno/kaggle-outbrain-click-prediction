#include "util/io.h"
#include "util/data.h"


std::vector<std::pair<std::string, std::string>> filesets {
    std::make_pair("cache/clicks_val_train.csv.gz", "cache/viewed_ads_val_train.csv.gz"),
    std::make_pair("cache/clicks_val_test.csv.gz", "cache/viewed_ads_val_test.csv.gz"),
    std::make_pair("../input/clicks_train.csv.gz", "cache/viewed_ads_full_train.csv.gz"),
    std::make_pair("../input/clicks_test.csv.gz", "cache/viewed_ads_full_test.csv.gz"),
};


std::streamsize buffer_size = 1024*1024;

std::pair<int, int> read_event_uid(const std::vector<std::string> & row) {
    return std::make_pair(stoi(row[0]), stoi(row[10]));
}


int main() {
    using namespace std;

    cout << "Loading reference data..." << endl;
    auto event_uids = read_vector("cache/events.csv.gz", read_event_uid, 23120127);

    for (uint ofs = 0; ofs < filesets.size(); ofs += 2) {
        cout << "Generating " << filesets[ofs].second << " and " << filesets[ofs+1].second << "... ";
        cout.flush();

        clock_t begin = clock();

        unordered_map<pair<int,int>, uint32_t> ad_view_counts;
        unordered_map<pair<int,int>, uint32_t> ad_click_counts;

        compressed_csv_file a_in(filesets[ofs].first);
        compressed_csv_file b_in(filesets[ofs+1].first);

        boost::iostreams::filtering_ostream a_out;
        a_out.push(boost::iostreams::gzip_compressor(), buffer_size, buffer_size);
        a_out.push(boost::iostreams::file_sink(filesets[ofs].second, std::ios_base::out | std::ios_base::binary), buffer_size, buffer_size);
        a_out << "ad_view_count,ad_click_count" << endl;

        boost::iostreams::filtering_ostream b_out;
        b_out.push(boost::iostreams::gzip_compressor(), buffer_size, buffer_size);
        b_out.push(boost::iostreams::file_sink(filesets[ofs+1].second, std::ios_base::out | std::ios_base::binary), buffer_size, buffer_size);
        b_out << "ad_view_count,ad_click_count" << endl;

        auto a_row = a_in.getrow();
        auto b_row = b_in.getrow();

        uint i = 0;

        while (!a_row.empty() || !b_row.empty()) {
            if (b_row.empty() || (!a_row.empty() && stoi(a_row[0]) < stoi(b_row[0]))) {
                auto uid = event_uids[stoi(a_row[0])];
                auto ad_id = stoi(a_row[1]);

                auto key = make_pair(uid, ad_id);

                a_out << ad_view_counts[key] << "," << ad_click_counts[key] << endl;

                ++ ad_view_counts[key];
                if (stoi(a_row[2]) > 0)
                    ++ ad_click_counts[key];

                a_row = a_in.getrow();
            } else {
                auto uid = event_uids[stoi(b_row[0])];
                auto ad_id = stoi(b_row[1]);

                auto key = make_pair(uid, ad_id);

                b_out << ad_view_counts[key] << "," << ad_click_counts[key] << endl;

                ++ ad_view_counts[key];

                b_row = b_in.getrow();
            }

            ++ i;

            if ( i % 5000000 == 0) {
                cout << (i / 1000000) << "M... ";
                cout.flush();
            }
        }

        clock_t end = clock();
        double elapsed = double(end - begin) / CLOCKS_PER_SEC;

        cout << "done in " << elapsed << " seconds" << endl;
    }

    cout << "Done." << endl;
}
