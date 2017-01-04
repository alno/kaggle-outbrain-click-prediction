#include "util/io.h"
#include "util/data.h"

std::vector<std::pair<std::string, std::vector<std::string>>> filesets {
    { "val", { "cache/clicks_val_train.csv.gz","cache/clicks_val_test.csv.gz" } },
    { "full", { "../input/clicks_train.csv.gz","../input/clicks_test.csv.gz" } }
};


void write_counts(const std::unordered_map<int, uint32_t> & map, const std::string & file_name) {
    using namespace std;

    cout << "Writing " << file_name << "... " << endl;

    ofstream outfile(file_name, std::ios_base::out | std::ios_base::binary);

    streamsize buffer_size = 1024*1024;
    boost::iostreams::filtering_streambuf<boost::iostreams::output> buf;
    buf.push(boost::iostreams::gzip_compressor(), buffer_size, buffer_size);
    buf.push(outfile, buffer_size, buffer_size);

    std::ostream out(&buf);

    out << "id,count" << endl;

    for (auto it = map.begin(); it != map.end(); ++ it)
        out << it->first << "," << it->second << endl;
}


int main() {
    using namespace std;

    cout << "Loading reference data..." << endl;
    auto data = load_reference_data();

    for (auto it = filesets.begin(); it != filesets.end(); ++ it) {
        unordered_map<int, uint32_t> ad_counts;
        unordered_map<int, uint32_t> ad_doc_counts;
        unordered_map<int, uint32_t> ev_doc_counts;


        cout << "Processing " << it->first << "... " << endl;

        for (auto fi = it->second.begin(); fi != it->second.end(); ++ fi) {
            clock_t begin = clock();

            cout << "  Loading " << *fi << "... ";
            cout.flush();

            compressed_csv_file file(*fi);

            for (int i = 0;; ++i) {
                auto row = file.getrow();

                if (row.empty())
                    break;

                if (i > 0 && i % 5000000 == 0) {
                    cout << (i / 1000000) << "M... ";
                    cout.flush();
                }

                // Extract fields
                int ev_id = stoi(row[0]);
                int ad_id = stoi(row[1]);

                auto ad = data.ads[ad_id];
                auto ev = data.events[ev_id];

                // Increment counters
                ++ ad_counts[ad_id];
                ++ ad_doc_counts[ad.document_id];
                ++ ev_doc_counts[ev.document_id];
            }

            clock_t end = clock();
            double elapsed = double(end - begin) / CLOCKS_PER_SEC;

            cout << "done in " << elapsed << " seconds" << endl;
        }

        write_counts(ad_counts, string("cache/ad_counts_") + it->first + string(".csv.gz"));
        write_counts(ad_doc_counts, string("cache/ad_doc_counts_") + it->first + string(".csv.gz"));
        write_counts(ev_doc_counts, string("cache/ev_doc_counts_") + it->first + string(".csv.gz"));
    }

    cout << "Done." << endl;
}
