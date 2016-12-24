#include "util/io.h"
#include "util/data.h"

std::vector<std::pair<std::string, std::string>> filesets = {
    std::make_pair("cache/clicks_val_train.csv.gz", "cache/val_train_vw.txt"),
    std::make_pair("cache/clicks_val_test.csv.gz", "cache/val_test_vw.txt"),
    std::make_pair("../input/clicks_train.csv.gz", "cache/full_train_vw.txt"),
    std::make_pair("../input/clicks_test.csv.gz", "cache/full_test_vw.txt"),
};

int main() {
    using namespace std;
    using namespace boost::iostreams;

    cout << "Loading reference data..." << endl;

    auto events = read_vector("cache/events.csv.gz", read_event, 23120127);
    auto ads = read_vector("../input/promoted_content.csv.gz", read_ad, 573099);
    auto documents = read_map("cache/documents.csv.gz", read_document);

    cout << "Generating files..." << endl;

    for (auto it = filesets.begin(); it != filesets.end(); ++ it) {
        auto in_file_name = it->first;
        auto out_file_name = it->second;

        cout << "  Generating " << out_file_name << "... ";
        cout.flush();

        clock_t begin = clock();

        compressed_csv_file in(in_file_name);
        ofstream out(out_file_name);

        for (int i = 0;; ++ i) {
            auto row = in.getrow();

            if (row.empty())
                break;

            int event_id = stoi(row[0]);
            int ad_id = stoi(row[1]);

            auto ad = ads[ad_id];
            auto event = events[event_id];

            stringstream line;

            line << "|a ad_" << ad_id << " p_" << event.platform << " ac_" << ad.campaign_id << " aa_" << ad.advertiser_id;
            line << "|l c_" << event.country << " s_" << event.state;
            line << "|d ad_d_" << ad.document_id << " d_" << event.document_id;
            line << endl;

            out << line.str();

            if (i > 0 && i % 5000000 == 0) {
                cout << (i / 1000000) << "M... ";
                cout.flush();
            }
        }

        clock_t end = clock();
        double elapsed = double(end - begin) / CLOCKS_PER_SEC;

        cout << "done in " << elapsed << " seconds" << endl;
    };

    cout << "Done." << endl;
}
