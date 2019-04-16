#include <cmath>
#include <algorithm>
#include <map>
#include <utility>
#include "simple-crf.hpp"

void SimpleCRFFrame::set_clusters(const Cluster* clusters) {
    std::copy(clusters, clusters + num_nodes, this->clusters.begin());
}

void SimpleCRFFrame::set_connectivity(const Connectivity* conn) {
    for (int i = 0; i < conn->num_nodes; i++) {
        for(int k = 0; k < conn->num_neighbors[i]; k++) {
            int j = conn->neighbors[i][k];
            edges.at(i).clear();
            edges[i].push_back(j);
        }
    }
}

void SimpleCRFFrame::normalize() {
    for (size_t i = 0; i < num_nodes; i++) {
        float sum = 0;
        for (size_t cls = 0; cls < num_classes; cls++) {
            sum += q[num_nodes * cls + i];
        }

        for (size_t cls = 0; cls < num_classes; cls++) {
            q[num_nodes * cls + i] /= sum;
        }
    }
}

void SimpleCRFFrame::set_unbiased() {
    float unary_const = logf(num_classes);
    std::fill_n(unaries.begin(), num_classes * num_nodes, unary_const);
}

void SimpleCRFFrame::set_mask(const int* classes, float confidence) {
    float lowest_proba = 1.0f / num_classes;
    float active_proba = lowest_proba + (1 - lowest_proba) * confidence;
    float inactive_proba = 1 - active_proba;
    float active_unary = -logf(active_proba), inactive_unary = -logf(inactive_proba);

    for (size_t i = 0; i < num_nodes; i++) {
        int active_cls = classes[i];

        for (size_t cls = 0; cls < num_classes; cls++) {
            unaries[num_nodes * cls + i] = ((int)cls == active_cls)? active_unary : inactive_unary;
        }
    }
}


void SimpleCRFFrame::set_proba(const float* probas) {
    for (size_t cls = 0; cls < num_classes; cls++) {
        for (size_t i = 0; i < num_nodes; i++) {
            unaries[num_nodes * cls + i] = -logf(probas[cls * num_nodes + i]);
        }
    }
}

void SimpleCRFFrame::reset_inferred() {
    for (size_t cls = 0; cls < num_classes; cls++) {
        for (size_t i = 0; i < num_nodes; i++) {
            q[num_nodes * cls + i] = expf(-unaries[cls * num_nodes + i]);
        }
    }
}

static inline float pow2(float a) {
    return a * a;
}

float SimpleCRFFrame::calc_temporal_pairwise_energy(int node, const SimpleCRFFrame& other) const {
    const Cluster &cluster_1 = clusters[node];
    const Cluster &cluster_2 = other.clusters[node];
    float exponent = -(
        pow2((cluster_1.r - cluster_2.r) / parent.params.temporal_srgb) +
        pow2((cluster_1.g - cluster_2.g) / parent.params.temporal_srgb) +
        pow2((cluster_1.b - cluster_2.b) / parent.params.temporal_srgb)
    ) / 2;
    return parent.params.temporal_w * expf(exponent);
}

float SimpleCRFFrame::calc_spatial_pairwise_energy(int node_i, int node_j) const {
    const Cluster &cluster_1 = clusters[node_i];
    const Cluster &cluster_2 = clusters[node_j];
    float exponent = -(
        pow2((cluster_1.r - cluster_2.r) / parent.params.temporal_srgb) +
        pow2((cluster_1.g - cluster_2.g) / parent.params.temporal_srgb) +
        pow2((cluster_1.b - cluster_2.b) / parent.params.temporal_srgb)
    ) / 2;
    return parent.params.spatial_w * expf(exponent);
}

void SimpleCRF::infer_once() {
    simple_crf_time_t first_time = get_first_time(), last_time = get_last_time();

    std::map<int, float *> new_qs;
    for (int t = first_time; t <= last_time; t++) {
        float *messages = new float[num_classes * num_nodes];
        float *compat_exps = new float[num_classes * num_nodes];

        const SimpleCRFFrame& frame = get_frame(t);
        // Message passing
        for (size_t cls = 0; cls < num_classes; cls++) {
            for (size_t i = 0; i < num_nodes; i++) {
                auto & neighbors = frame.connected_nodes(i);
                // pairwise literal
                float message = 0;
                for (auto j : neighbors) {
                    float pair_q = frame.q[num_nodes * cls  + j];
                    // 1. spatial pairwise energy
                    float spatial_energy = frame.calc_spatial_pairwise_energy(i, j);
                    message += spatial_energy * pair_q;
                }

                if (t > first_time) {
                    const SimpleCRFFrame& prev_frame = get_frame(t - 1);
                    prev_frame.q[num_nodes * cls + i];
                    message += frame.calc_temporal_pairwise_energy(i, prev_frame) * prev_frame.q[num_nodes * cls + i];
                }

                if (t < last_time) {
                    const SimpleCRFFrame& next_frame = get_frame(t + 1);
                    next_frame.q[num_nodes * cls + i];
                    message += frame.calc_temporal_pairwise_energy(i, next_frame) * next_frame.q[num_nodes * cls + i];
                }
                messages[cls * num_nodes + i] = message;
            }
        }
        for (size_t cls = 0; cls < num_classes; cls++) {
            for (size_t i = 0; i < num_nodes; i++) {
                float gathtered_message_sum = 0;
                for (size_t other_cls = 0 ; other_cls < num_classes; other_cls++) {
                    if (other_cls == cls) continue; // Potts model
                    gathtered_message_sum += compat_by_class[other_cls] * messages[other_cls * num_nodes + i];
                }
                compat_exps[cls * num_nodes + i] = expf(-(frame.unaries[cls * num_nodes + i] + gathtered_message_sum));
             }
        }

        for (size_t i = 0; i < num_nodes; i++) {
            float sum = 0;
            for (size_t cls = 0; cls < num_classes; cls++) {
                sum += compat_exps[num_nodes * cls + i];
            }
            for (size_t cls = 0; cls < num_classes; cls++) {
                compat_exps[num_nodes * cls + i] /= sum;
            }
        }
        new_qs[t] = compat_exps;
        delete [] messages;
    }

    for (int t = first_time; t <= last_time; t++) {
        SimpleCRFFrame& frame = get_frame(t);
        float *new_q = new_qs[t];
        std::copy(new_q, new_q + (num_nodes * num_classes), frame.q.begin());
        delete [] new_q;
    }
}

void SimpleCRF::inference(size_t max_iter) {
    for (size_t i = 0; i < max_iter; i++) {
        infer_once();
    }
}


// C Api
extern "C" {
    simple_crf_t simple_crf_new(size_t num_classes, size_t num_nodes) { return new SimpleCRF(num_classes, num_nodes); }
    void simple_crf_free(simple_crf_t crf) { delete crf; };

    SimpleCRFParams simple_crf_get_params(simple_crf_t crf) { return crf->params; }
    void simple_crf_set_params(simple_crf_t crf, SimpleCRFParams params) { crf->params = params; }
    void simple_crf_set_compat(simple_crf_t crf, int cls, float compat_value) { crf->compat_by_class[cls] = compat_value; }
    float simple_crf_get_compat(simple_crf_t crf, int cls) { return crf->compat_by_class[cls]; }

    simple_crf_time_t simple_crf_first_time(simple_crf_t crf) { return crf->get_first_time(); }
    simple_crf_time_t simple_crf_last_time(simple_crf_t crf) { return crf->get_last_time(); }
    size_t simple_crf_num_time_frames(simple_crf_t crf) { return crf->num_frames(); }
    simple_crf_time_t simple_crf_pop_time_frame(simple_crf_t crf) { return crf->pop_frame(); }
    simple_crf_frame_t simple_crf_push_time_frame(simple_crf_t crf) { return &crf->push_frame(); };
    simple_crf_frame_t simple_crf_time_frame(simple_crf_t crf, simple_crf_time_t time) {
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
    void simple_crf_frame_set_mask(simple_crf_frame_t frame, const int* classes, float confidence) {
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

    struct _conn_iter {
        simple_crf_frame_t frame;
        size_t i;
        size_t j;
    };

    simple_crf_conn_iter_t simple_crf_frame_pairwise_connection(simple_crf_frame_t frame, int node_i) {
        _conn_iter *iter = new _conn_iter();
        iter->frame = frame;
        iter->i = node_i;
        iter->j = 0;
        return iter;
    }

    simple_crf_conn_iter_t simple_crf_frame_pairwise_connection_next(simple_crf_conn_iter_t iter, int *node_j) {
        _conn_iter *conn_iter = (_conn_iter * )iter;
        simple_crf_frame_t frame = conn_iter->frame;

        auto& connected_nodes = frame->connected_nodes(conn_iter->i);
        if (conn_iter->j < connected_nodes.size()) {
            *node_j = connected_nodes[conn_iter->j++];
            return conn_iter;
        } else {
            return nullptr;
        }
    }

    void simple_crf_frame_pairwise_connection_end(simple_crf_conn_iter_t iter) {
        delete static_cast<_conn_iter *>(iter);
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
    void simple_crf_inference(simple_crf_t crf, size_t max_iter) {
        return crf->inference(max_iter);
    }

    /*
     * Utils
     */
    simple_crf_t simple_crf_copy(simple_crf_t crf) {
        return new SimpleCRF(*crf);
    }
};
