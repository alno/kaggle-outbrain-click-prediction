#include "util/io.h"
#include "util/data.h"


std::vector<std::pair<std::string, std::string>> filesets {
    { "cache/clicks_cv1_train.csv.gz", "cv1_train" },
    { "cache/clicks_cv1_test.csv.gz", "cv1_test" },
    { "cache/clicks_cv2_train.csv.gz", "cv2_train" },
    { "cache/clicks_cv2_test.csv.gz", "cv2_test" },
    { "../input/clicks_train.csv.gz", "full_train" },
    { "../input/clicks_test.csv.gz", "full_test" },
};

std::vector<int> max_timestamps = {
    1123200000,
    1123200000,
    std::numeric_limits<int>::max()
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
std::unordered_multimap<int, std::pair<int, float>> document_categories;
std::unordered_multimap<int, std::pair<int, float>> document_topics;

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


class doc_category_writer {
    std::unordered_map<std::pair<int, int>, float> category_views_map;
public:
    std::string get_header() {
        return "category_view_weight";
    }

    void prepare(int uid, int document_id, int timestamp) {
        auto doc_categories = document_categories.equal_range(document_id);

        for (auto it = doc_categories.first; it != doc_categories.second; ++ it)
            category_views_map[std::make_pair(it->second.first, uid)] = 0;
    }

    void update(int uid, int document_id, int timestamp) {
        auto doc_categories = document_categories.equal_range(document_id);

        for (auto it = doc_categories.first; it != doc_categories.second; ++ it) {
            auto cv_it = category_views_map.find(std::make_pair(it->second.first, uid));
            if (cv_it != category_views_map.end())
                cv_it->second += it->second.second;
        }
    }

    void write(std::ostream & out, int uid, int document_id, int timestamp) {
        auto doc_categories = document_categories.equal_range(document_id);

        float category_view_weight = 0;

        for (auto it = doc_categories.first; it != doc_categories.second; ++ it)
            category_view_weight += category_views_map[std::make_pair(it->second.first, uid)];

        out << category_view_weight << std::endl;
    }
};


class doc_topic_writer {
    std::unordered_map<std::pair<int, int>, float> topic_views_map;
public:
    std::string get_header() {
        return "topic_view_weight";
    }

    void prepare(int uid, int document_id, int timestamp) {
        auto doc_topics = document_topics.equal_range(document_id);

        for (auto it = doc_topics.first; it != doc_topics.second; ++ it)
            topic_views_map[std::make_pair(it->second.first, uid)] = 0;
    }

    void update(int uid, int document_id, int timestamp) {
        auto doc_topics = document_topics.equal_range(document_id);

        for (auto it = doc_topics.first; it != doc_topics.second; ++ it) {
            auto cv_it = topic_views_map.find(std::make_pair(it->second.first, uid));
            if (cv_it != topic_views_map.end())
                cv_it->second += it->second.second;
        }
    }

    void write(std::ostream & out, int uid, int document_id, int timestamp) {
        auto doc_topics = document_topics.equal_range(document_id);

        float topic_view_weight = 0;

        for (auto it = doc_topics.first; it != doc_topics.second; ++ it)
            topic_view_weight += topic_views_map[std::make_pair(it->second.first, uid)];

        out << topic_view_weight << std::endl;
    }
};


template <typename W>
void generate(const std::string & file_name_prefix, uint ofs) {
    using namespace std;

    cout << "Generating " << file_name_prefix << "..." << endl;

    W w;

    cout << "  Loading click data..." << endl;
    for (uint fi = ofs; fi < ofs + 2; ++ fi) {
        auto in_file_name = filesets[fi].first;

        cout << "    Loading " << in_file_name << "... ";
        cout.flush();

        time_t begin = time(nullptr);

        compressed_csv_file file(in_file_name);

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

        cout << "done in " << (time(nullptr) - begin) << " seconds" << endl;
    }

    {
        cout << "  Processing page views data... ";
        cout.flush();

        time_t begin = time(nullptr);

        compressed_csv_file file("../input/page_views.csv.gz");
        int found = 0;
        int max_timestamp = max_timestamps[ofs / 2];

        for (int i = 0;; ++i) {
            auto row = file.getrow();

            if (row.empty())
                break;

            auto uuid = row[0];
            auto document_id = stoi(row[1]);
            auto timestamp = stoi(row[2]);

            if (timestamp <= max_timestamp) {
                // Register view
                auto uid_it = uuid_map.find(uuid);
                if (uid_it != uuid_map.end()) {
                    w.update(uid_it->second, document_id, timestamp);
                }
            }

            if (i > 0 && i % 5000000 == 0) {
                cout << (i / 1000000) << "M... ";
                cout.flush();
            }
        }

        cout << "done in " << (time(nullptr) - begin) << " seconds, found " << found << " entries" << endl;
    }

    cout << "  Generating viewed docs features..." << endl;
    for (uint fi = ofs; fi < ofs + 2; ++ fi) {
        auto out_file_name = string("cache/") + file_name_prefix + string("_") + filesets[fi].second + string(".csv.gz");

        cout << "  Generating " << out_file_name << "... ";
        cout.flush();

        time_t begin = time(nullptr);

        compressed_csv_file file(filesets[fi].first);

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

        cout << "done in " << (time(nullptr) - begin) << " seconds" << endl;
    }
}


int main() {
    using namespace std;

    cout << "Loading reference data..." << endl;
    events = read_vector("cache/events.csv.gz", read_event_info, 23120127);
    ad_doc_ids = read_vector("../input/promoted_content.csv.gz", read_ad_document, 573099);
    documents = read_map("cache/documents.csv.gz", read_document);
    document_categories = read_multi_map("../input/documents_categories.csv.gz", read_document_annotation);
    document_topics = read_multi_map("../input/documents_topics.csv.gz", read_document_annotation);

    for (uint ofs = 0; ofs < filesets.size(); ofs += 2) {
        generate<doc_source_writer>("viewed_docs", ofs);
        generate<doc_category_writer>("viewed_categories", ofs);
        generate<doc_topic_writer>("viewed_topics", ofs);
    }

    cout << "Done." << endl;
}
