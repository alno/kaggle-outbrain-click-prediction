#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <ctime>

#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <boost/type_index.hpp>


void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


class compressed_csv_file {
public:
    boost::iostreams::filtering_istream in;
    std::vector<std::string> header;
public:
    compressed_csv_file(const std::string & name) {
        std::streamsize buffer_size = 1024*1024;

        in.push(boost::iostreams::gzip_decompressor(), buffer_size, buffer_size);
        in.push(boost::iostreams::file_source(name, std::ios_base::in | std::ios_base::binary), buffer_size, buffer_size);

        header = getrow();
    }

    std::string getline() {
        std::string line;
        std::getline(in, line);
        return line;
    }

    std::vector<std::string> getrow() {
        return split(getline(), ',');
    }

    operator bool() {
        return !in.eof();
    }
};


template <typename K, typename T>
std::unordered_map<K, T> read_map(const std::string & file_name, std::pair<K, T> read_entry(const std::vector<std::string> &)) {
    using namespace std;

    clock_t begin = clock();

    cout << "  Loading " << boost::typeindex::type_id<T>().pretty_name() << "s... ";
    cout.flush();

    compressed_csv_file file(file_name);
    unordered_map<K, T> res;

    for (int i = 0;; ++i) {
        vector<string> row = file.getrow();

        if (row.empty())
            break;

        res.insert(read_entry(row));

        if (i > 0 && i % 5000000 == 0) {
            cout << (i / 1000000) << "M... ";
            cout.flush();
        }
    }

    clock_t end = clock();
    double elapsed = double(end - begin) / CLOCKS_PER_SEC;

    cout << "done in " << elapsed << " seconds, " << res.size() << " records." << endl;

    return res;
}


template <typename K, typename T>
std::vector<T> read_vector(const std::string & file_name, std::pair<K, T> read_entry(const std::vector<std::string> &), size_t size) {
    using namespace std;

    clock_t begin = clock();

    cout << "  Loading " << boost::typeindex::type_id<T>().pretty_name() << "s... ";
    cout.flush();

    compressed_csv_file file(file_name);
    vector<T> res;

    // Resize vector to contain all elements
    res.resize(size);

    for (int i = 0;; ++i) {
        vector<string> row = file.getrow();

        if (row.empty())
            break;

        auto entry = read_entry(row);

        res[entry.first] = move(entry.second);

        if (i > 0 && i % 5000000 == 0) {
            cout << (i / 1000000) << "M... ";
            cout.flush();
        }
    }

    clock_t end = clock();
    double elapsed = double(end - begin) / CLOCKS_PER_SEC;

    cout << "done in " << elapsed << " seconds." << endl;

    return res;
}


template <typename K, typename T>
std::unordered_multimap<K, T> read_multi_map(const std::string & file_name, std::pair<K, T> read_entry(const std::vector<std::string> &)) {
    using namespace std;

    clock_t begin = clock();

    cout << "  Loading " << boost::typeindex::type_id<T>().pretty_name() << "s... ";
    cout.flush();

    compressed_csv_file file(file_name);
    unordered_multimap<K, T> res;

    for (int i = 0;; ++i) {
        vector<string> row = file.getrow();

        if (row.empty())
            break;

        res.insert(read_entry(row));

        if (i > 0 && i % 5000000 == 0) {
            cout << (i / 1000000) << "M... ";
            cout.flush();
        }
    }

    clock_t end = clock();
    double elapsed = double(end - begin) / CLOCKS_PER_SEC;

    cout << "done in " << elapsed << " seconds, " << res.size() << " records." << endl;

    return res;
}
