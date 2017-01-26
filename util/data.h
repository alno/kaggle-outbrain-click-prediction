#pragma once

#include <string>
#include <vector>
#include <stdexcept>

#include "io.h"

struct event {
    uint32_t document_id;
    uint32_t uid;
    int64_t timestamp;
    uint16_t platform;
    uint16_t weekday;
    uint16_t hour;
    std::string country;
    std::string state;
    std::string region;
    std::string location;
};

struct ad {
    uint32_t document_id;
    uint32_t campaign_id;
    uint32_t advertiser_id;
};

struct document {
    uint32_t source_id;
    uint32_t publisher_id;
    int64_t publish_timestamp;
};


struct traffic_source_id_list {
    std::vector<uint> internal, social, search;
};



// Small util


namespace std {
    template <typename A, typename B>
    struct hash<std::pair<A, B>> {
        std::size_t operator()(const std::pair<A, B>& k) const {
          return std::hash<A>()(k.first) ^ (std::hash<B>()(k.second) >> 1);
        }
    };
}



// Functions to read data types


std::pair<int, event> read_event(const std::vector<std::string> & row) {
    event e;

    e.document_id = stoi(row[2]);
    e.timestamp = stoll(row[3]);

    try {
        e.platform = stoi(row[4]);
    } catch (std::invalid_argument) {
        e.platform = 0;
    }

    e.location = row[5];

    e.country = row[6];
    e.state = row[7];
    e.region = row[8];

    e.hour = stoi(row[9]);
    e.weekday = stoi(row[10]);
    e.uid = stoi(row[11]);

    return std::make_pair(stoi(row[0]), e);
}

std::pair<int, ad> read_ad(const std::vector<std::string> & row) {
    ad a;

    a.document_id = stoi(row[1]);
    a.campaign_id = stoi(row[2]);
    a.advertiser_id = stoi(row[3]);

    return std::make_pair(stoi(row[0]), a);
}

std::pair<int, document> read_document(const std::vector<std::string> & row) {
    document d;

    d.source_id = stoi(row[1]);
    d.publisher_id = stoi(row[2]);
    d.publish_timestamp = stoll(row[4]);

    return std::make_pair(stoi(row[0]), d);
}

std::pair<int, std::pair<int, float>> read_document_annotation(const std::vector<std::string> & row) {
    return std::make_pair(stoi(row[0]), std::make_pair(stoi(row[1]), stof(row[2])));
}

std::pair<int, int> read_count(const std::vector<std::string> & row) {
    return std::make_pair(stoi(row[0]), stoi(row[1]));
}

std::vector<uint> parse_id_list(const std::string & field) {
    using namespace std;

    vector<uint> values;

    stringstream ss;
    ss.str(field);

    string item;
    while (getline(ss, item, ' ')) {
        values.push_back(stoi(item));
    }

    return values;
}

std::pair<std::pair<uint, uint>, std::vector<uint>> read_display_indexed_id_list(const std::vector<std::string> & row) {
    using namespace std;

    auto key = make_pair(stoi(row[0]), stoi(row[1]));
    auto val = parse_id_list(row[2]);

    return make_pair(key, val);
}


std::pair<uint, std::vector<uint>> read_uid_indexed_id_list(const std::vector<std::string> & row) {
    using namespace std;

    return make_pair(stoi(row[0]), parse_id_list(row[1]));
}


std::pair<uint, traffic_source_id_list> read_uid_indexed_trfsrc_id_list(const std::vector<std::string> & row) {
    using namespace std;

    traffic_source_id_list trf;
    trf.internal = parse_id_list(row[1]);
    trf.social = parse_id_list(row[2]);

    if (row.size() > 3)
        trf.search = parse_id_list(row[3]);

    return make_pair(stoi(row[0]), move(trf));
}

std::vector<event> read_events() {
    return read_vector("cache/events.csv.gz", read_event, 23120127);
}

std::vector<ad> read_ads() {
    return read_vector("../input/promoted_content.csv.gz", read_ad, 573099);
}



// All data


struct reference_data {
    std::vector<event> events;
    std::vector<ad> ads;
    std::unordered_map<int, document> documents;
    std::unordered_multimap<int, std::pair<int, float>> document_categories;
    std::unordered_multimap<int, std::pair<int, float>> document_topics;
    std::unordered_multimap<int, std::pair<int, float>> document_entities;

    std::unordered_map<std::pair<uint, uint>, std::vector<uint>> viewed_docs_one_hour_after;
    //std::unordered_map<std::pair<uint, uint>, std::vector<uint>> viewed_docs_six_hours_after;

    std::unordered_map<uint, std::vector<uint>> doc_ad_others;
    std::unordered_map<uint, std::vector<uint>> viewed_doc_trf_source;
    std::unordered_map<uint, std::vector<uint>> viewed_doc_sources;

    std::unordered_map<uint, traffic_source_id_list> viewed_trfsrc_doc_sources;
    std::unordered_map<uint, traffic_source_id_list> viewed_trfsrc_docs;
};

reference_data load_reference_data() {
    reference_data res;
    res.viewed_trfsrc_doc_sources = read_map("cache/viewed_trfsrc_doc_sources.csv.gz", read_uid_indexed_trfsrc_id_list);
    res.viewed_trfsrc_docs = read_map("cache/viewed_trfsrc_docs.csv.gz", read_uid_indexed_trfsrc_id_list);

    res.events = read_events();
    res.ads = read_ads();
    res.documents = read_map("cache/documents.csv.gz", read_document);
    res.document_categories = read_multi_map("../input/documents_categories.csv.gz", read_document_annotation);
    res.document_topics = read_multi_map("../input/documents_topics.csv.gz", read_document_annotation);
    res.document_entities = read_multi_map("cache/documents_entities.csv.gz", read_document_annotation);

    res.viewed_docs_one_hour_after = read_map("cache/viewed_docs_one_hour_after.csv.gz", read_display_indexed_id_list);
    //res.viewed_docs_six_hours_after = read_map("cache/viewed_docs_six_hours_after.csv.gz", read_display_indexed_id_list);

    res.doc_ad_others = read_map("cache/doc_ad_others.csv.gz", read_uid_indexed_id_list);
    res.viewed_doc_trf_source = read_map("cache/viewed_doc_trf_source.csv.gz", read_uid_indexed_id_list);
    res.viewed_doc_sources = read_map("cache/viewed_doc_sources.csv.gz", read_uid_indexed_id_list);


    return res;
}
