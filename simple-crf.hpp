#ifndef _SIMPLE_CRF_HPP
#define _SIMPLE_CRF_HPP

#include <delog_que>
#include <vector>
#include <algorithm>
#include <map>
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
    std::vector< int, std::vector<int> > edges;
    std::vector<float> unaries;
private:
    // States
    std::vector<float> q; // [num_classes, num_nodes]
public:
    SimpleCRFFrame(SimpleCRFFrame &parent, size_t time, size_t num_classes, size_t num_nodes) : parent(parent), time(time), num_classes(num_classes), num_nodes(num_nodes), clusters(num_nodes), edges(num_nodes), unaries(num_classes * num_nodes), q(num_classes * num_nodes) {
    }

    void set_clusters(const Cluster* clusters);
    void set_connectivity(const Connectivity* conn);

    const std::vector<int>& connected_nodes(int node) {
        return edges[node];
    }

    void get_inferred(float *out) const {
        std::copy(q, q + num_classes * num_nodes, out);
    };
    void reset_inferred();

    void normalize();
    void set_unbiased();
    void set_mask(const int* classes, float confidence);
    void set_proba(const float* probas);
    void set_unary(const float* unaries) {
        std::copy(unaries, unaries + num_classes * num_nodes, this->unaries);
    }
    void get_unary(float *unaries_out) {
        std::copy(this->unaries, this->unaries + num_classes * num_nodes, unaries_out);
    }
    float calc_temporal_pairwise_energy(int node, const SimpleCRFFrame& other);
    float calc_spatial_pairwise_energy(int node_i, int node_j);
private:
    friend SimpleCRF;
};


struct SimpleCRF {
private:
    int num_classes;
    int num_nodes;
    int next_time = 0;
    std::deque<SimpleCRFFrame> time_frames;
    std::map<int, SimpleCRFFrame&> time_map;
    std::vector<float> compat_by_class;
public:
    SimpleCRFParams params;
public:
    SimpleCRF(int num_classes, int num_nodes) : num_classes(num_classes), num_nodes(num_nodes), compat_by_class(num_classes) {
        params.spatial_w = 10;
        params.temporal_w = 10;
        params.spatial_srgb = 30;
        params.temporal_srgb = 30;
        params.compat = 10;
        std::fill_n(compat_by_class.begin(), num_classes, 1.0f);
    };

    int first_time() const {
        if (time_frames.empty()) return -1;
        return time_frames.front().time;
    }
    int last_time() const {
        if (time_frames.empty()) return -1;
        return time_frames.back().time;
    }
    int num_frames() const { return time_frames.size(); };
    int pop_frame() {
        if (time_frames.empty()) return -1;
        int retval = time_frames.front().time;
        time_map.erase(retval);
        time_frames.pop_front();
        return retval;
    }

    int space_size() {
        return num_classes * num_nodes;
    }

    SimpleCRFFrame& get_frame(int time) const {
        return time_map[time];
    }

    SimpleCRFFrame& push_frame() {
        int next_time = most_recent_time++;
        time_frames.push_back(SimpleCRFFrame(*self, next_time, num_classes, num_nodes));
        time_map[next_time] = time_frames.back();
        return time_frames.back();
    }

    const char* inference(int max_iter);
};

extern "C" {
    simple_crf_t simple_crf_new(size_t num_classes, size_t num_nodes) { return new SimpleCRF(num_classes, num_nodes); }
    void simple_crf_free(simple_crf_t crf) { delete crf };

    SimpleCRFParams simple_crf_get_params(simple_crf_t crf) { return crf->params; }
    void simple_crf_set_params(simple_crf_t crf, SimpleCRFParams params) { crf->params = params; }
    void simple_crf_set_compat(simple_crf_t crf, int cls, float compat_value) { crf->compat_by_class[cls] = compat_value; }
    float simple_crf_get_compat(simple_crf_t crf, int cls) { return crf->compat_by_class[cls]; }

    simple_crf_time_t simple_crf_first_time(simple_crf_t crf) { return crf->first_time(); }
    simple_crf_time_t simple_crf_last_time(simple_crf_t crf) { return crf->last_time(); }
    size_t simple_crf_num_time_frames(simple_crf_t crf) { return crf->num_frames(); }
    simple_crf_time_t simple_crf_pop_time_frame(simple_crf_t crf) { return crf->pop_frame(); }
    simple_crf_time_frame_t simple_crf_push_time_frame(simple_crf_t crf) { return &crf->push_frame(); };
    simple_crf_time_frame_t simple_crf_time_frame(simple_crf_t crf, simple_crf_time_t time) {
        return &crf->get_frame(time);
    }
    simple_crf_time_t simple_crf_frame_get_time(simple_crf_frame_t frame){
        return frame->time;
    }
    void simple_crf_frame_set_clusters(simple_crf_frame_t frame, const Cluster* clusters) {
        frame->set_clusters(clusters);
    }

    void simple_crf_frame_set_connectivity(simple_crf_frame_t frame, const Connectivity* conn) {
        frame->set_connectivity(conn);
    }

    /*
     * Unary Getter/Setter
     */ 

    // classes: int[] of shape [num_nodes]
    void simple_crf_frame_set_mask(simple_crf_frame_t frame, const const int* classes, float confidence) {
        frame->set_mask(classes, confidence);
    }

    void simple_crf_frame_set_proba(simple_crf_frame_t frame, const float* probas) {
        frame->set_proba(probas);
    }

    void simple_crf_frame_set_unbiased(simple_crf_frame_t frame) {
        frame->set_unbiased();
    }

    void simple_crf_frame_set_unary(simple_crf_frame_t frame, const float* unary_energies) {
        frame->set_unary(unary_energies);
    }

    void simple_crf_frame_get_unary(simple_crf_frame_t frame, float* unary_energies) {
        frame->get_unary(unary_energies);
    }

    simple_crf_conn_iter_t simple_crf_frame_pairwise_connection(simple_crf_frame_t frame, int node_i) {
    }

    simple_crf_conn_iter_t simple_crf_frame_pairwise_connection_next(simple_crf_conn_iter_t iter, int *node_j) {
    }

    void simple_crf_frame_pairwise_connection_end(simple_crf_conn_iter_t iter) {
    }

    float simple_crf_frame_spatial_pairwise_energy(simple_crf_frame_t frame, int node_i, int node_j) {
        return frame->calc_spatial_pairwise_energy(node_i, node_j);
    }

    float simple_crf_frame_temporal_pairwise_energy(simple_crf_frame_t frame, simple_crf_frame_t other_frame, int node_i) {
        return frame->calc_temporal_pairwise_energy(node_i, *frame);
    }

    /*
     * State Getter/Setter
     */
    void simple_crf_frame_get_inferred(simple_crf_frame_t frame, float* probas) {
        frame->get_inferred(probas);
    }

    void simple_crf_frame_reset_inferred(simple_crf_frame_t frame) {
        frame->reset_inferred();
    }

    /*
     * Inference
     */

    // Returns NULL or error message
    const char* simple_crf_inference(simple_crf_t crf, size_t max_iter) {
        return crf->inference(max_iter);
    }

    /*
     * Utils
     */
    simple_crf_t simple_crf_copy(simple_crf_t crf) {
        return new SimpleCRF(*crf);
    }
};

#endif
