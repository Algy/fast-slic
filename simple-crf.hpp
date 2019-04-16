#ifndef _SIMPLE_CRF_HPP
#define _SIMPLE_CRF_HPP

#include <deque>
#include <vector>
#include <algorithm>
#include <map>
#include <functional>
#include "simple-crf.h"

class SimpleCRFFrame {
private:
    SimpleCRF& parent;
public:
    simple_crf_time_t time;
    size_t num_classes;
    size_t num_nodes;
private:
    std::vector<Cluster> clusters;
    std::vector< std::vector<int> > edges;
    std::vector<float> unaries;
private:
    // States
    std::vector<float> q; // [num_classes, num_nodes]
public:
    SimpleCRFFrame(SimpleCRF &parent, size_t time, size_t num_classes, size_t num_nodes) : parent(parent), time(time), num_classes(num_classes), num_nodes(num_nodes), clusters(num_nodes), edges(num_nodes, std::vector<int>(12)), unaries(num_classes * num_nodes), q(num_classes * num_nodes) {
    }

    void set_clusters(const Cluster* clusters);
    void set_connectivity(const Connectivity* conn);

    const std::vector<int>& connected_nodes(int node) const {
        return edges[node];
    }

    void get_inferred(float *out) const { std::copy(q.begin(), q.end(), out); };
    void reset_inferred();

    void normalize();
    void set_unbiased();
    void set_mask(const int* classes, float confidence);
    void set_proba(const float* probas);
    void set_unary(const float* unaries) {
        std::copy(unaries, unaries + (num_classes * num_nodes), this->unaries.begin());
    }
    void get_unary(float *unaries_out) const {
        std::copy(this->unaries.begin(), this->unaries.end(), unaries_out);
    }
    float calc_temporal_pairwise_energy(int node, const SimpleCRFFrame& other) const;
    float calc_spatial_pairwise_energy(int node_i, int node_j) const;
private:
    friend SimpleCRF;
};


struct SimpleCRF {
private:
    size_t num_classes;
    size_t num_nodes;
    simple_crf_time_t next_time = 0;
    std::deque<SimpleCRFFrame> time_frames;
    std::map<simple_crf_time_t, SimpleCRFFrame*> time_map;
public:
    std::vector<float> compat_by_class;
    SimpleCRFParams params;
public:
    SimpleCRF(size_t num_classes, size_t num_nodes) : num_classes(num_classes), num_nodes(num_nodes), compat_by_class(num_classes) {
        params.spatial_w = 10;
        params.temporal_w = 10;
        params.spatial_srgb = 30;
        params.temporal_srgb = 30;
        std::fill_n(compat_by_class.begin(), num_classes, 1.0f);
    };

    simple_crf_time_t get_first_time() const {
        if (time_frames.empty()) return -1;
        return time_frames.front().time;
    }

    simple_crf_time_t get_last_time() const {
        if (time_frames.empty()) return -1;
        return time_frames.back().time;
    }

    size_t num_frames() const { return time_frames.size(); };

    simple_crf_time_t pop_frame() {
        if (time_frames.empty()) return -1;
        int retval = time_frames.front().time;
        time_map.erase(retval);
        time_frames.pop_front();
        return retval;
    }

    size_t space_size() const {
        return num_classes * num_nodes;
    }

    SimpleCRFFrame& get_frame(simple_crf_time_t time) {
        SimpleCRFFrame& retval = *time_map[time];
        return retval;
    }

    SimpleCRFFrame& push_frame() {
        simple_crf_time_t next_time = this->next_time++;
        time_frames.push_back(SimpleCRFFrame(*this, next_time, num_classes, num_nodes));
        time_map[next_time] = &time_frames.back();
        return time_frames.back();
    }

    void inference(size_t max_iter);
private:
    void infer_once();
};

#endif
