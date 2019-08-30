#ifndef _SIMPLE_CRF_HPP
#define _SIMPLE_CRF_HPP

#include <deque>
#include <vector>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <functional>
#include "simple-crf.h"

static inline float pow2(float a) { return a * a; }

struct SimpleCRFFrame {
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
    SimpleCRFFrame(SimpleCRF &parent, size_t time, size_t num_classes, size_t num_nodes) : parent(parent), time(time), num_classes(num_classes), num_nodes(num_nodes), clusters(num_nodes), edges(num_nodes), unaries(space_size()), q(space_size()) {
        for (auto& cluster : clusters) {
            cluster.num_members = 1;
        }
    }


    void get_clusters(Cluster* clusters) {
        std::copy(this->clusters.begin(), this->clusters.end(), clusters);
    }
    void set_clusters(const Cluster* clusters);
    void set_connectivity(const Connectivity* conn);

    const std::vector<int>& connected_nodes(int node) const {
        return edges.at(node);
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

    size_t space_size() const { return num_classes * num_nodes; }

    void get_unary(float *unaries_out) const {
        std::copy(this->unaries.begin(), this->unaries.end(), unaries_out);
    }
    float inline calc_temporal_pairwise_energy(int node, const SimpleCRFFrame& other) const;
    float inline calc_spatial_pairwise_energy(int node_i, int node_j) const;
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
        params.spatial_srgb = 13;
        params.temporal_srgb = 13;
        params.spatial_sxy = 80;
        params.spatial_smooth_w = 0;
        params.spatial_smooth_sxy = 3;
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

    size_t get_num_frames() const { return time_frames.size(); };

    simple_crf_time_t pop_frame() {
        if (time_frames.empty()) return -1;
        int retval = time_frames.front().time;
        time_map.erase(retval);
        time_frames.pop_front();
        return retval;
    }

    SimpleCRFFrame& get_frame(simple_crf_time_t time) {
        SimpleCRFFrame* ptr = time_map[time];
        if (ptr == nullptr) {
            throw std::out_of_range("Time out of range");
        }

        SimpleCRFFrame& retval = *ptr;
        return retval;
    }

    SimpleCRFFrame& push_frame() {
        simple_crf_time_t next_time = this->next_time++;
        time_frames.push_back(SimpleCRFFrame(*this, next_time, num_classes, num_nodes));
        time_map[next_time] = &time_frames.back();
        return time_frames.back();
    }

    size_t space_size() const { return num_classes * num_nodes; }
    void initialize();
    void inference(size_t max_iter);
private:
    void infer_once();
};

float inline SimpleCRFFrame::calc_temporal_pairwise_energy(int node, const SimpleCRFFrame& other) const {
    if (this == &other) return 0;
    const Cluster &cluster_1 = clusters[node];
    const Cluster &cluster_2 = other.clusters[node];
    float stdev = parent.params.temporal_srgb;
    float weight = parent.params.temporal_w;
    float exponent = -(
        pow2((cluster_1.r - cluster_2.r) / stdev) +
        pow2((cluster_1.g - cluster_2.g) / stdev) +
        pow2((cluster_1.b - cluster_2.b) / stdev)
    ) / 2.0f;
    return weight * expf(exponent);
}

float inline SimpleCRFFrame::calc_spatial_pairwise_energy(int node_i, int node_j) const {
    if (node_i == node_j) return 0;

    const Cluster &cluster_1 = clusters[node_i];
    const Cluster &cluster_2 = clusters[node_j];
    float stdev = parent.params.spatial_srgb, weight = parent.params.spatial_w;
    float sxy = parent.params.spatial_sxy;
    float smooth_weight = parent.params.spatial_smooth_w;
    float smooth_sxy = parent.params.spatial_smooth_sxy;
    float exponent = -(
        pow2((cluster_1.r - cluster_2.r) / stdev) +
        pow2((cluster_1.g - cluster_2.g) / stdev) +
        pow2((cluster_1.b - cluster_2.b) / stdev)
    ) / 2.0f;

    exponent += -(
        pow2((cluster_1.x - cluster_2.x) / sxy) + 
        pow2((cluster_1.y - cluster_2.y) / sxy)
    ) / 2.0f;

    float smooth_exponent = -(
        pow2((cluster_1.x - cluster_2.x) / smooth_sxy) + 
        pow2((cluster_1.y - cluster_2.y) / smooth_sxy)
    ) / 2.0f;
    return weight * expf(exponent) + smooth_weight * expf(smooth_exponent);
}

#endif

