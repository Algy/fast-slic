#ifndef _SIMPLE_CRF_HPP
#define _SIMPLE_CRF_HPP

#include <delog_que>
#include <vector>
#include <algorithm>
#include <map>
#include "simple-crf.h"

class CRFPairwise {
public:
    int from;
    int to;
    float spatial_energy_cache;
public:
    CRFPairwiseEnergy(int from, int to) : from(from), to(to), spatial_energy_cache(0);
};

class SimpleCRFFrame {
public:
    simple_crf_time_t time;
    size_t num_classes;
    size_t num_nodes;
private:
    // Configurations
    bool unary_configured = false;
    bool pixel_configured = false;
    bool pairwise_configured = false;

    std::vector<CRFPixel> pixels;
    std::vector< int, std::vector<CRFPairwise> > edges;
    std::vector<float> unaries;
private:
    // States
    std::vector<float> log_q; // [num_classes, num_nodes]
public:
    SimpleCRFFrame(int time, int num_classes, int num_nodes) : time(time), num_classes(num_classes), num_nodes(num_nodes) {
        pixels.reserve(num_classes);
        unaries.reserve(num_classes);
        log_q.reserve(num_classes * num_nodes);
    }

    void set_clusters(const Cluster* clusters);
    void set_connectivity(const Connectivity* conn);

    const float* get_log_proba() const { return log_q; };
    bool ready() { return unary_configured && pairwise_configured && pixel_configured };

    void normalize();
    void set_unbiased();
    void set_mask(const int* classes, float confidence);
    void set_log_proba(const float* probas);
    void reset_frame_state();
    void set_pixels(SimpleCRFFrame& frame, const uint8_t* yxrgbs);
    void set_connectivity(simple_crf_t crf, int t, const SimpleCRFConnectivity* conn);
private:
    float calc_temporal_pairwise_energy(const SimpleCRFParams& params, int node, const SimpleCRFFrame& other);
    float calc_spatial_pairwise_energy(const SimpleCRFParams& params, int node_i, int node_j);
};

struct SimpleCRF {
private:
    int num_classes;
    int num_nodes;
    int next_time = 0;
    std::deque<SimpleCRFFrame> time_frames;
    std::map<int, SimpleCRFFrame&> time_map;
public:
    SimpleCRFParams params;
public:
    SimpleCRF(int num_classes, int num_nodes) : num_classes(num_classes), num_nodes(num_nodes) {
        params.spatial_w = 10;
        params.temporal_w = 10;
        params.spatial_srgb = 30;
        params.temporal_srgb = 30;
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

    SimpleCRFFrame& get_frame(int time) const {
        return time_map[time];
    }

    int push_frame() {
        int next_time = most_recent_time++;
        time_frames.push_back(SimpleCRFFrame(next_time, num_classes, num_nodes));
        time_map[next_time] = time_frames.back();
        return next_time;
    }

    const char* iterate(int max_iter);
};

extern "C" {
    simple_crf_t simple_crf_new(int num_classes, int num_nodes) { return new SimpleCRF(); };
    void simple_crf_free(simple_crf_t crf) { delete crf; };

    SimpleCRFParams simple_crf_get_params(simple_crf_t crf) { return crf->params; }
    void simple_crf_set_params(simple_crf_t crf, SimpleCRFParams params) {
        crf->params = params;
    }

    int simple_crf_first_time(simple_crf_t crf) { return crf->first_time(); }
    int simple_crf_last_time(simple_crf_t crf) { return crf->last_time() }
    int simple_crf_num_time_frames(simple_crf_t crf) { return crf->num_frames() }
    int simple_crf_pop_time_frame(simple_crf_t crf) { return crf->pop_frame(); }
    int simple_crf_push_time_frame(simple_crf_t crf) { return crf->push_frame(); }

    void simple_crf_set_pixels(simple_crf_t crf, int t, const uint8_t* yxrgbs) {
        crf->get_frame(t).set_pixels(pixels);
    }

    void simple_crf_set_connectivity(simple_crf_t crf, int t, const SimpleCRFConnectivity* conn) {
        crf->get_frame(t).set_connectivity(conn);
    }

    // classes: int[] of shape [num_nodes]
    void simple_crf_set_frame_mask(simple_crf_t crf, int t, const int* classes, float confidence) { crf->get_frame(t).set_mask(classes, confidence); };
    // probas: float[] of shape [num_classes, num_nodes]
    void simple_crf_set_frame_log_proba(simple_crf_t crf, int t, const float* log_probas) { crf->get_frame(t).set_log_proba(log_probas); }
    void simple_crf_set_frame_unbiased(simple_crf_t crf, int t) { crf->get_frame(t).set_unbiased(); }
    const char* simple_crf_iterate(simple_crf_t crf, int max_iter) { return crf->iterate(max_iter); }
    void simple_crf_reset_frame_state(simple_crf_t crf, int t) { crf->get_frame(t).reset_frame_state(); };

    // out_log_probas: flaot[] of shape [num_classes, num_nodes]
    void simple_crf_get_log_proba(simple_crf_t crf, int t, float* out_log_probas) {
        const float* proba = crf->get_log_proba();
        std::copy(proba, proba + num_classes * num_nodes, out_log_probas);
    }

    simple_crf_t simple_crf_copy(simple_crf_t crf) {
        return new SimpleCRF(*crf);
    }
};

#endif
