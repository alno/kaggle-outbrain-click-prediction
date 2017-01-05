#include "util/io.h"
#include "util/data.h"


std::vector<std::pair<std::string, std::string>> filesets {
    std::make_pair("cache/clicks_val_train.csv.gz", "cache/viewed_docs_val_train.csv.gz"),
    std::make_pair("cache/clicks_val_test.csv.gz", "cache/viewed_docs_val_test.csv.gz"),
    std::make_pair("../input/clicks_train.csv.gz", "cache/viewed_docs_full_train.csv.gz"),
    std::make_pair("../input/clicks_test.csv.gz", "cache/viewed_docs_full_test.csv.gz"),
};

struct event_info {
    int uid;
    int timestamp;
};


std::unordered_map<std::string, int> uuid_map;
std::unordered_map<std::pair<int, int>, int> publisher_views_map;
std::unordered_map<std::pair<int, int>, int> source_views_map;


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
    return std::make_pair(stoi(row[0]), stoi(row[1]));
}


int main() {
    using namespace std;

    cout << "Loading reference data..." << endl;
    auto events = read_vector("cache/events.csv.gz", read_event_info, 23120127);
    auto ad_documents = read_vector("../input/promoted_content.csv.gz", read_ad_document, 573099);
    auto documents = read_map("cache/documents.csv.gz", read_document);


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
            auto document_id = ad_documents.at(stoi(row[1]));
            auto document = documents.at(document_id);

            if (document.publisher_id > 0)
                publisher_views_map[make_pair(document.publisher_id, ev.uid)] = 0;

            if (document.source_id > 0)
                source_views_map[make_pair(document.source_id, ev.uid)] = 0;

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

            auto uuid = row[0];
            auto document_id = stoi(row[1]);
            auto document = documents.at(document_id);

            // Register view
            auto uid_it = uuid_map.find(uuid);
            if (uid_it != uuid_map.end()) {
                auto uid = uid_it->second;

                auto pv_it = publisher_views_map.find(make_pair(document.publisher_id, uid));
                if (pv_it != publisher_views_map.end())
                    pv_it->second ++;

                auto sv_it = source_views_map.find(make_pair(document.source_id, uid));
                if (sv_it != source_views_map.end())
                    sv_it->second ++;
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

    cout << "Generating viewed docs features..." << endl;
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

        out << "publisher_views,source_views" << endl;

        for (int i = 0;; ++i) {
            auto row = file.getrow();

            if (row.empty())
                break;

            auto ev = events.at(stoi(row[0]));
            auto document_id = ad_documents.at(stoi(row[1]));
            auto document = documents.at(document_id);

            auto publisher_view_times = document.publisher_id > 0 ? publisher_views_map[make_pair(document.publisher_id, ev.uid)] : -1;
            auto source_view_times = document.source_id > 0 ? source_views_map[make_pair(document.source_id, ev.uid)] : -1;

            out << publisher_view_times << ","
                << source_view_times << endl;

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
