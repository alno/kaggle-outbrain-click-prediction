#include "util/io.h"

std::vector<std::pair<std::string, std::string>> filesets {
    std::make_pair("cache/clicks_cv1_train.csv.gz", "cache/similarity_cv1_train.csv.gz"),
    std::make_pair("cache/clicks_cv1_test.csv.gz", "cache/similarity_cv1_test.csv.gz"),
    std::make_pair("cache/clicks_cv2_train.csv.gz", "cache/similarity_cv2_train.csv.gz"),
    std::make_pair("cache/clicks_cv2_test.csv.gz", "cache/similarity_cv2_test.csv.gz"),
    std::make_pair("../input/clicks_train.csv.gz", "cache/similarity_full_train.csv.gz"),
    std::make_pair("../input/clicks_test.csv.gz", "cache/similarity_full_test.csv.gz"),
};

struct event_info {
    int uid;
    int document_id;
};


std::vector<int> event_documents;
std::vector<int> ad_documents;

typedef std::pair<int, float> annotation;

std::unordered_map<int,std::vector<annotation>> document_categories_map;
std::unordered_map<int,std::vector<annotation>> document_topics_map;
std::unordered_map<int,std::vector<annotation>> document_entities_map;

void create_document_entries(int document_id) {
    if (document_categories_map.count(document_id) == 0)
        document_categories_map.insert(std::make_pair(document_id, std::vector<annotation>()));

    if (document_topics_map.count(document_id) == 0)
        document_topics_map.insert(std::make_pair(document_id, std::vector<annotation>()));

    if (document_entities_map.count(document_id) == 0)
        document_entities_map.insert(std::make_pair(document_id, std::vector<annotation>()));
}

std::pair<int, int> read_ad_document(const std::vector<std::string> & row) {
    int document_id = stoi(row[1]);

    create_document_entries(document_id);

    return std::make_pair(stoi(row[0]), document_id);
}

std::pair<int, int> read_event_document(const std::vector<std::string> & row) {
    int document_id = stoi(row[2]);

    create_document_entries(document_id);

    return std::make_pair(stoi(row[0]), document_id);
}

std::pair<int, annotation> read_document_annotation(const std::vector<std::string> & row) {
    return std::make_pair(stoi(row[0]), std::make_pair(stoi(row[1]), stof(row[2])));
}

struct similarity_values {
    float product;
    float jaccard;
};


/**
  Compute different similarity measures between two vectors, sorted by key
*/
similarity_values similarity(const std::vector<annotation> & va, const std::vector<annotation> & vb) {
    auto ia = va.begin();
    auto ib = vb.begin();

    float product = 0;

    int intersection_size = 0;
    int union_size = 0;

    while (ia != va.end() && ib != vb.end()) {
        if (ia->first < ib->first) {
            union_size ++;
            ia ++;
        } else if (ia->first > ib->first) {
            union_size ++;
            ib ++;
        } else {
            product += ia->second * ib->second;
            union_size ++;
            intersection_size ++;
            ia ++;
            ib ++;
        }
    }

    while (ia != va.end()) {
        union_size ++;
        ia ++;
    }

    while (ib != vb.end()) {
        union_size ++;
        ib ++;
    }

    similarity_values res;
    res.product = product;
    res.jaccard = union_size > 0 ? float(intersection_size) / union_size : 0;

    return res;
}


int main() {
    using namespace std;

    cout << "Loading reference data..." << endl;
    read_vector(event_documents, "cache/events.csv.gz", read_event_document, 23120127);
    read_vector(ad_documents, "../input/promoted_content.csv.gz", read_ad_document, 573099);

    read_sorted_vector_map(document_categories_map, "../input/documents_categories.csv.gz", read_document_annotation);
    read_sorted_vector_map(document_topics_map, "../input/documents_topics.csv.gz", read_document_annotation);
    read_sorted_vector_map(document_entities_map, "cache/documents_entities.csv.gz", read_document_annotation);


    cout << "Generating similarity features..." << endl;
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

        out << "cat_sim_product,cat_sim_jaccard,top_sim_product,top_sim_jaccard,ent_sim_product,ent_sim_jaccard" << endl;

        for (int i = 0;; ++i) {
            auto row = file.getrow();

            if (row.empty())
                break;

            auto ev_doc_id = event_documents.at(stoi(row[0]));
            auto ad_doc_id = ad_documents.at(stoi(row[1]));

            // Compute document similarity metrics
            auto cat_sim = similarity(document_categories_map.at(ad_doc_id), document_categories_map.at(ev_doc_id));
            auto top_sim = similarity(document_topics_map.at(ad_doc_id), document_topics_map.at(ev_doc_id));
            auto ent_sim = similarity(document_entities_map.at(ad_doc_id), document_entities_map.at(ev_doc_id));

            // Write similarity metrics
            out << cat_sim.product << "," << cat_sim.jaccard << ",";
            out << top_sim.product << "," << top_sim.jaccard << ",";
            out << ent_sim.product << "," << ent_sim.jaccard << endl;

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
