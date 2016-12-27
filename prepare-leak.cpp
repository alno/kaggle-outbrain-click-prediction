#include "util/io.h"


namespace std {
    template <typename A, typename B>
    struct hash<std::pair<A, B>> {
        std::size_t operator()(const std::pair<A, B>& k) const {
          return std::hash<A>()(k.first) ^ (std::hash<B>()(k.second) >> 1);
        }
    };
}

//

std::vector<std::pair<std::string, std::string>> filesets {
    std::make_pair("../input/clicks_train.csv.gz", "cache/leak_full_train.csv.gz"),
    std::make_pair("../input/clicks_test.csv.gz", "cache/leak_full_test.csv.gz"),
};

struct event_info {
    int uid;
    int timestamp;
};


std::unordered_map<std::string, int> uuid_map;
std::unordered_map<std::pair<int, int>, std::vector<int>> views_map;
std::unordered_map<int,int> documents_map;


std::pair<int, event_info> read_event_info(const std::vector<std::string> & row) {
    auto id = stoi(row[0]);
    auto uuid = row[1];

    event_info res;
    res.timestamp = stoi(row[3]);

    auto it = uuid_map.find(uuid);

    if (it == uuid_map.end()) {
        res.uid = uuid_map.size();
        uuid_map.insert(std::make_pair(uuid, res.uid));
    } else {
        res.uid = it->second;
    }

    return std::make_pair(id, res);
}

std::pair<int, int> read_ad_document(const std::vector<std::string> & row) {
    int ad_document_id = stoi(row[1]);

    if (documents_map.count(ad_document_id) == 0)
        documents_map.insert(std::make_pair(ad_document_id, 0));

    return std::make_pair(stoi(row[0]), ad_document_id);
}


int main() {
    using namespace std;

    cout << "Loading reference data..." << endl;
    auto events = read_vector("cache/events.csv.gz", read_event_info, 23120127);
    auto ad_documents = read_vector("../input/promoted_content.csv.gz", read_ad_document, 573099);


    cout << "Loading click data..." << endl;
    for (auto it = filesets.begin(); it != filesets.end(); ++ it) {
        cout << "  Loading " << it->first << "... ";
        cout.flush();

        clock_t begin = clock();

        compressed_csv_file file(it->first);

        for (int i = 0;; ++i) {
            auto row = file.getrow();

            if (row.empty())
                break;

            auto ev = events.at(stoi(row[0]));
            auto ad_document_id = ad_documents.at(stoi(row[1]));

            views_map.insert(make_pair(make_pair(ad_document_id, ev.uid), std::vector<int>()));

            if (i > 0 && i % 5000000 == 0) {
                cout << (i / 1000000) << "M... ";
                cout.flush();
            }
        }

        clock_t end = clock();
        double elapsed = double(end - begin) / CLOCKS_PER_SEC;

        cout << "done in " << elapsed << " seconds" << endl;
    }

    {
        cout << "Processing leak data... ";
        cout.flush();

        clock_t begin = clock();

        compressed_csv_file file("../input/page_views.csv.gz");
        int found = 0;

        for (int i = 0;; ++i) {
            auto row = file.getrow();

            if (row.empty())
                break;

            auto document_id = stoi(row[1]);
            auto uuid = row[0];

            // Register view
            auto uid_it = uuid_map.find(uuid);
            if (uid_it != uuid_map.end()) {
                auto key = make_pair(document_id, uid_it->second);
                auto it = views_map.find(key);

                if (it != views_map.end()) {
                    it->second.push_back(stoi(row[2]));
                    found ++;
                }
            }

            // Register document view
            auto doc_it = documents_map.find(document_id);
            if (doc_it != documents_map.end()) {
                doc_it->second ++;
            }

            if (i > 0 && i % 5000000 == 0) {
                cout << (i / 1000000) << "M... ";
                cout.flush();
            }
        }

        clock_t end = clock();
        double elapsed = double(end - begin) / CLOCKS_PER_SEC;

        cout << "done in " << elapsed << " seconds, found " << found << " entries" << endl;
    }

    cout << "Generating leak features..." << endl;
    for (auto it = filesets.begin(); it != filesets.end(); ++ it) {
        cout << "  Generating " << it->second << "... ";
        cout.flush();

        clock_t begin = clock();

        compressed_csv_file file(it->first);
        ofstream outfile(it->second, std::ios_base::out | std::ios_base::binary);

        streamsize buffer_size = 1024*1024;
        boost::iostreams::filtering_streambuf<boost::iostreams::output> buf;
        buf.push(boost::iostreams::gzip_compressor(), buffer_size, buffer_size);
        buf.push(outfile, buffer_size, buffer_size);

        std::ostream out(&buf);

        out << "viewed,not_viewed" << endl;

        for (int i = 0;; ++i) {
            auto row = file.getrow();

            if (row.empty())
                break;

            auto ev = events.at(stoi(row[0]));
            auto ad_document_id = ad_documents.at(stoi(row[1]));

            auto view_times = views_map.at(make_pair(ad_document_id, ev.uid));
            auto doc_views = documents_map.at(ad_document_id);

            auto viewed = view_times.size() > 0;

            out << int(viewed) << "," << int(!viewed && (doc_views > 0)) << endl;

            if (i > 0 && i % 5000000 == 0) {
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
