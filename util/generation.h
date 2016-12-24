#pragma once

#include "io.h"

template <typename D>
void generate_files(const D & data, const std::vector<std::pair<std::string, std::string>> & filesets, std::string encode_row_fun(const D &, int, int, int)) {
    using namespace std;
    using namespace boost::iostreams;

    for (auto it = filesets.begin(); it != filesets.end(); ++ it) {
        auto in_file_name = it->first;
        auto out_file_name = it->second;

        cout << "  Generating " << out_file_name << "... ";
        cout.flush();

        clock_t begin = clock();

        compressed_csv_file in(in_file_name);
        ofstream out(out_file_name);

        bool label = in.header.size() == 3;

        for (int i = 0;; ++ i) {
            auto row = in.getrow();

            if (row.empty())
                break;

            out << encode_row_fun(data, stoi(row[0]), stoi(row[1]), label ? stoi(row[2]) : -1);

            if (i > 0 && i % 5000000 == 0) {
                cout << (i / 1000000) << "M... ";
                cout.flush();
            }
        }

        clock_t end = clock();
        double elapsed = double(end - begin) / CLOCKS_PER_SEC;

        cout << "done in " << elapsed << " seconds" << endl;
    };
}
