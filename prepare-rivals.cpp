#include "util/io.h"

std::vector<std::pair<std::string, std::string>> filesets {
    { "cache/clicks_cv1_train.csv.gz", "cv1_train" },
    { "cache/clicks_cv1_test.csv.gz", "cv1_test" },
    { "cache/clicks_cv2_train.csv.gz", "cv2_train" },
    { "cache/clicks_cv2_test.csv.gz", "cv2_test" },
    { "../input/clicks_train.csv.gz", "full_train" },
    { "../input/clicks_test.csv.gz", "full_test" },
};


void write_rivals(std::ostream & out, const std::vector<uint> rivals) {
    for (uint i = 0; i < rivals.size(); ++ i) {
        out << rivals.size() << ",";

        uint w = 0;
        for (uint j = 0; j < rivals.size(); ++ j) {
            if (i != j) {
                if (w > 0)
                    out << " ";

                out << rivals[j];

                w ++;
            }
        }

        out << "," << i << std::endl;
    }
}


int main() {
    using namespace std;

    cout << "Generating rivals features..." << endl;
    for (auto it = filesets.begin(); it != filesets.end(); ++ it) {
        string out_file_name = string("cache/rivals_") + it->second + string(".csv.gz");

        cout << "  Generating " << out_file_name << "... ";
        cout.flush();

        clock_t begin = clock();

        compressed_csv_file file(it->first);
        ofstream outfile(out_file_name, std::ios_base::out | std::ios_base::binary);

        streamsize buffer_size = 1024*1024;
        boost::iostreams::filtering_streambuf<boost::iostreams::output> buf;
        buf.push(boost::iostreams::gzip_compressor(), buffer_size, buffer_size);
        buf.push(outfile, buffer_size, buffer_size);

        std::ostream out(&buf);

        out << "rival_count,rivals,rank" << endl;

        uint cur_group_id = 0;
        std::vector<uint> cur_rivals;

        for (int i = 0;; ++i) {
            auto row = file.getrow();

            if (row.empty())
                break;

            uint group_id = stoi(row[0]);
            uint ad_id = stoi(row[1]);

            if (cur_group_id != group_id) {
                write_rivals(out, cur_rivals);

                cur_group_id = group_id;
                cur_rivals.clear();
            }

            cur_rivals.push_back(ad_id);

            if (i > 0 && i % 5000000 == 0) {
                cout << (i / 1000000) << "M... ";
                cout.flush();
            }
        }

        write_rivals(out, cur_rivals);

        clock_t end = clock();
        double elapsed = double(end - begin) / CLOCKS_PER_SEC;

        cout << "done in " << elapsed << " seconds" << endl;
    }

    cout << "Done." << endl;
}
