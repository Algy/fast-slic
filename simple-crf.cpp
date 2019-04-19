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
        edges.at(i).clear();
        for(int k = 0; k < conn->num_neighbors[i]; k++) {
            int j = conn->neighbors[i][k];
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
    float unary_const = logf((float)num_classes);
    std::fill(unaries.begin(), unaries.end(), unary_const);
}

void SimpleCRFFrame::set_mask(const int* classes, float confidence) {
    float lowest_proba = 1.0f / num_classes;
    float active_proba = lowest_proba + (1 - lowest_proba) * confidence;
    float inactive_proba = (1 - active_proba) / (float)(num_classes - 1);
    float active_unary = -logf(active_proba), inactive_unary = -logf(inactive_proba);

    std::fill(unaries.begin(), unaries.end(), inactive_unary);
    for (size_t i = 0; i < num_nodes; i++) {
        int active_cls = classes[i];
        unaries[num_nodes * active_cls + i] = active_unary;
    }
}


void SimpleCRFFrame::set_proba(const float* probas) {
    std::transform(probas, probas + space_size(), unaries.begin(), [](float proba) -> float { return -logf(proba); });
}

void SimpleCRFFrame::reset_inferred() {
    std::transform(unaries.begin(), unaries.end(), q.begin(), [](float unary) -> float { return expf(-unary); });
}


void SimpleCRF::infer_once() {
    simple_crf_time_t first_time = get_first_time(), last_time = get_last_time();
    std::vector<float *> new_probas;

    for (simple_crf_time_t t = first_time; t <= last_time; t++) {
        float *messages = new float[space_size()];
        float *compat_exps = new float[space_size()];
        const SimpleCRFFrame& frame = get_frame(t);

        // Message passing
        for (size_t cls = 0; cls < num_classes; cls++) {
            for (size_t i = 0; i < num_nodes; i++) {
                const std::vector<int>& neighbors = frame.connected_nodes(i);

                int num_members = frame.clusters[i].num_members;
                if (num_members <= 0) {
                    num_members = 1;
                }

                // sum of energy from neighbors
                float message = 0;
                for (auto neighbor : neighbors) {
                    float neighbor_q = frame.q[num_nodes * cls + neighbor];
                    // 1. spatial pairwise energy
                    float spatial_energy = frame.calc_spatial_pairwise_energy(neighbor, i);
                    message += spatial_energy * neighbor_q * sqrtf((float)frame.clusters[neighbor].num_members / num_members);
                }

                if (t > first_time) {
                    const SimpleCRFFrame& prev_frame = get_frame(t - 1);
                    message += frame.calc_temporal_pairwise_energy(i, prev_frame) * prev_frame.q[num_nodes * cls + i] * sqrtf((float)prev_frame.clusters[i].num_members / num_members);
                }

                if (t < last_time) {
                    const SimpleCRFFrame& next_frame = get_frame(t + 1);
                    message += frame.calc_temporal_pairwise_energy(i, next_frame) * next_frame.q[num_nodes * cls + i] * sqrtf((float)next_frame.clusters[i].num_members / num_members);
;
                }
                messages[cls * num_nodes + i] = message;
            }
        }

        // Compatibility transform
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

        // Normalize
        float *sums = new float[num_nodes];
        std::fill_n(sums, num_nodes, 0.0f);
        for (size_t cls = 0; cls < num_classes; cls++) {
            for (size_t i = 0; i < num_nodes; i++) {
                sums[i] += compat_exps[num_nodes * cls + i];
            }
        }

        for (size_t i = 0; i < num_nodes; i++) {
            sums[i] = (sums[i] < 1e-5)? 1e-5 : sums[i];
        }

        for (size_t cls = 0; cls < num_classes; cls++) {
            for (size_t i = 0; i < num_nodes; i++) {
                compat_exps[num_nodes * cls + i] /= sums[i];
            }
        }

        new_probas.push_back(compat_exps);
        delete [] sums;
        delete [] messages;
    }

    size_t iter = 0;
    for (simple_crf_time_t t = first_time; t <= last_time; t++) {
        SimpleCRFFrame& frame = get_frame(t);
        float *new_q = new_probas.at(iter++);
        std::copy(new_q, new_q + space_size(), frame.q.begin());
    }

    // Clean up
    for (float* buf : new_probas) {
        delete [] buf;
    }
}

void SimpleCRF::initialize() {
    for (auto& time_frame : time_frames) {
        time_frame.reset_inferred();
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
    void simple_crf_initialize(simple_crf_t crf) {
        crf->initialize();
    }
    void simple_crf_free(simple_crf_t crf) { delete crf; };

    SimpleCRFParams simple_crf_get_params(simple_crf_t crf) { return crf->params; }
    void simple_crf_set_params(simple_crf_t crf, SimpleCRFParams params) { crf->params = params; }
    void simple_crf_set_compat(simple_crf_t crf, int cls, float compat_value) { crf->compat_by_class[cls] = compat_value; }
    float simple_crf_get_compat(simple_crf_t crf, int cls) { return crf->compat_by_class[cls]; }

    simple_crf_time_t simple_crf_first_time(simple_crf_t crf) { return crf->get_first_time(); }
    simple_crf_time_t simple_crf_last_time(simple_crf_t crf) { return crf->get_last_time(); }
    size_t simple_crf_num_time_frames(simple_crf_t crf) { return crf->get_num_frames(); }
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
