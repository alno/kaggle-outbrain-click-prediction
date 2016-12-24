#pragma once

#include <string>
#include <vector>
#include <stdexcept>

struct event {
    int document_id;
    int timestamp;
    int platform;
    std::string uuid;
    std::string country;
    std::string state;
    std::string region;
};

struct ad {
    int document_id;
    int campaign_id;
    int advertiser_id;
};

struct document {
    int source_id;
    int publisher_id;
};


std::pair<int, event> read_event(const std::vector<std::string> & row) {
    event e;

    e.document_id = stoi(row[2]);
    e.timestamp = stoi(row[3]);

    try {
        e.platform = stoi(row[4]);
    } catch (std::invalid_argument) {
        e.platform = 0;
    }

    e.uuid = row[1];
    e.country = row[5];
    e.state = row[6];
    e.region = row[7];

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

    return std::make_pair(stoi(row[0]), d);
}
