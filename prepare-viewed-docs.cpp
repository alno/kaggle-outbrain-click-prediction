#include "util/io.h"
#include "util/data.h"


std::vector<std::pair<std::string, std::string>> filesets {
    std::make_pair("cache/clicks_val_train.csv.gz", "val_train"),
    std::make_pair("cache/clicks_val_test.csv.gz", "val_test"),
    std::make_pair("../input/clicks_train.csv.gz", "full_train"),
    std::make_pair("../input/clicks_test.csv.gz", "full_test"),
};

struct event_info {
    int uid;
    int timestamp;
};


std::unordered_map<std::string, int> uuid_map;


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


std::vector<event_info> events;
std::vector<int> ad_doc_ids;
std::unordered_map<int, document> documents;

std::streamsize buffer_size = 1024*1024;


class doc_source_writer {
    std::unordered_map<std::pair<int, int>, int> publisher_views_map;
    std::unordered_map<std::pair<int, int>, int> source_views_map;
public:
    std::string get_header() {
        return "publisher_view_count,source_view_count";
    }

    void prepare(int uid, int document_id, int timestamp) {
        using namespace std;

        auto document = documents.at(document_id);

        if (document.publisher_id > 0)
            publisher_views_map[make_pair(document.publisher_id, uid)] = 0;

        if (document.source_id > 0)
            source_views_map[make_pair(document.source_id, uid)] = 0;
    }

    void update(int uid, int document_id, int timestamp) {
        using namespace std;

        auto document = documents.at(document_id);

        auto pv_it = publisher_views_map.find(make_pair(document.publisher_id, uid));
        if (pv_it != publisher_views_map.end())
            pv_it->second ++;

        auto sv_it = source_views_map.find(make_pair(document.source_id, uid));
        if (sv_it != source_views_map.end())
            sv_it->second ++;
    }

    void write(std::ostream & out, int uid, int document_id, int timestamp) {
        using namespace std;

        auto document = documents.at(document_id);

        auto publisher_view_times = document.publisher_id > 0 ? publisher_views_map[make_pair(document.publisher_id, uid)] : -1;
        auto source_view_times = document.source_id > 0 ? source_views_map[make_pair(document.source_id, uid)] : -1;

        out << publisher_view_times << ","
            << source_view_times << endl;
    }
};


template <typename W>
void generate(const std::string & file_name_prefix) {
    using namespace std;

    cout << "Generating " << file_name_prefix << "..." << endl;

    W w;

    cout << "  Loading click data..." << endl;
    for (auto it = filesets.begin(); it != filesets.end(); ++ it) {
        cout << "    Loading " << it->first << "... ";
        cout.flush();

        clock_t begin = clock();

        compressed_csv_file file(it->first);

        for (int i = 0;; ++i) {
            auto row = file.getrow();

            if (row.empty())
                break;

            auto ev = events.at(stoi(row[0]));
            auto document_id = ad_doc_ids.at(stoi(row[1]));

            w.prepare(ev.uid, document_id, ev.timestamp);

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
        cout << "  Processing page views data... ";
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
            auto timestamp = stoi(row[2]);

            // Register view
            auto uid_it = uuid_map.find(uuid);
            if (uid_it != uuid_map.end()) {
                w.update(uid_it->second, document_id, timestamp);
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

    cout << "  Generating viewed docs features..." << endl;
    for (auto it = filesets.begin(); it != filesets.end(); ++ it) {
        auto out_file_name = string("cache/") + file_name_prefix + it->second + string(".csv.gz");

        cout << "  Generating " << out_file_name << "... ";
        cout.flush();

        clock_t begin = clock();

        compressed_csv_file file(it->first);

        boost::iostreams::filtering_ostream out;
        out.push(boost::iostreams::gzip_compressor(), buffer_size, buffer_size);
        out.push(boost::iostreams::file_sink(out_file_name, std::ios_base::out | std::ios_base::binary), buffer_size, buffer_size);

        out << w.get_header() << endl;

        for (int i = 0;; ++i) {
            auto row = file.getrow();

            if (row.empty())
                break;

            auto ev = events.at(stoi(row[0]));
            auto document_id = ad_doc_ids.at(stoi(row[1]));

            w.write(out, ev.uid, document_id, ev.timestamp);

            if (i > 0 && i % 5000000 == 0) {
                cout << (i / 1000000) << "M... ";
                cout.flush();
            }
        }

        clock_t end = clock();
        double elapsed = double(end - begin) / CLOCKS_PER_SEC;

        cout << "done in " << elapsed << " seconds" << endl;
    }
}


int main() {
    using namespace std;

    cout << "Loading reference data..." << endl;
    events = read_vector("cache/events.csv.gz", read_event_info, 23120127);
    ad_doc_ids = read_vector("../input/promoted_content.csv.gz", read_ad_document, 573099);
    documents = read_map("cache/documents.csv.gz", read_document);

    generate<doc_source_writer>("viewed_docs");

    cout << "Done." << endl;
}
