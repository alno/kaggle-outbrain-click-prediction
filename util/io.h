#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
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

        if (!in)
            throw std::runtime_error(std::string("Can't open file ") + name);
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

    time_t begin = time(nullptr);

    cout << "  Loading " << boost::typeindex::type_id<T>().pretty_name() << "s from " << file_name << "... ";
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

    cout << "done in " << (time(nullptr) - begin) << " seconds, " << res.size() << " records." << endl;

    return res;
}


template <typename K, typename T>
void read_vector(std::vector<T> & res, const std::string & file_name, std::pair<K, T> read_entry(const std::vector<std::string> &), size_t size) {
    using namespace std;

    time_t begin = time(nullptr);

    cout << "  Loading " << boost::typeindex::type_id<T>().pretty_name() << "s from " << file_name << "... ";
    cout.flush();

    compressed_csv_file file(file_name);

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

    cout << "done in " << (time(nullptr) - begin) << " seconds." << endl;
}


template <typename K, typename T>
std::vector<T> read_vector(const std::string & file_name, std::pair<K, T> read_entry(const std::vector<std::string> &), size_t size) {
    std::vector<T> res;
    read_vector(res, file_name, read_entry, size);
    return res;
}


template <typename K, typename T>
std::unordered_multimap<K, T> read_multi_map(const std::string & file_name, std::pair<K, T> read_entry(const std::vector<std::string> &)) {
    using namespace std;

    time_t begin = time(nullptr);

    cout << "  Loading " << boost::typeindex::type_id<T>().pretty_name() << "s from " << file_name << "... ";
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

    cout << "done in " << (time(nullptr) - begin) << " seconds, " << res.size() << " records." << endl;

    return res;
}


template <typename K, typename T>
void read_sorted_vector_map(std::unordered_map<K, std::vector<T>> & res, const std::string & file_name, std::pair<K, T> read_entry(const std::vector<std::string> &)) {
    using namespace std;

    time_t begin = time(nullptr);

    cout << "  Loading " << boost::typeindex::type_id<T>().pretty_name() << "s from " << file_name << "... ";
    cout.flush();

    compressed_csv_file file(file_name);

    for (int i = 0;; ++i) {
        vector<string> row = file.getrow();

        if (row.empty())
            break;

        auto e = read_entry(row);

        if (res.count(e.first) == 0)
            res.insert(std::make_pair(e.first, vector<T>()));

        res.at(e.first).push_back(move(e.second));

        if (i > 0 && i % 5000000 == 0) {
            cout << (i / 1000000) << "M... ";
            cout.flush();
        }
    }

    for (auto it = res.begin(); it != res.end(); ++ it)
        std::sort(it->second.begin(), it->second.end());

    cout << "done in " << (time(nullptr) - begin) << " seconds, " << res.size() << " records." << endl;
}

template <typename K, typename T>
std::unordered_map<K, std::vector<T>> read_sorted_vector_map(const std::string & file_name, std::pair<K, T> read_entry(const std::vector<std::string> &)) {
    std::unordered_map<K, std::vector<T>> res;
    read_sorted_vector_map(res, file_name, read_entry);
    return res;
}
