#pragma once

#include "io.h"

template <typename D, typename W>
void generate_files(const D & data, const std::vector<std::pair<std::vector<std::string>, std::string>> & filesets) {
    using namespace std;
    using namespace boost::iostreams;

    for (auto it = filesets.begin(); it != filesets.end(); ++ it) {
        auto in_file_name = it->first[0];
        auto out_file_name = it->second;

        cout << "  Generating " << out_file_name << "... ";
        cout.flush();

        clock_t begin = clock();

        vector<unique_ptr<compressed_csv_file>> in_files;

        for (auto in_it = it->first.begin(); in_it != it->first.end(); ++in_it)
            in_files.push_back(unique_ptr<compressed_csv_file>(new compressed_csv_file(*in_it)));

        W writer(out_file_name);

        for (int i = 0;; ++ i) {
            vector<vector<string>> rows;

            for (auto in_it = in_files.begin(); in_it != in_files.end(); ++in_it)
                rows.push_back((*in_it)->getrow());

            if (rows[0].empty())
                break;

            writer.write(data, rows);

            if (i > 0 && i % 5000000 == 0) {
                cout << (i / 1000000) << "M... ";
                cout.flush();
            }
        }

        writer.finish();

        clock_t end = clock();
        double elapsed = double(end - begin) / CLOCKS_PER_SEC;

        cout << "done in " << elapsed << " seconds" << endl;
    };
}
