#include <cmath>
#include <algorithm>
#include <map>
#include "simple-crf.hpp"

void SimpleCRFFrame::set_clusters(const Cluster* clusters) {
    std::copy(clusters, clusters + num_nodes, clusters.begin());
}

void SimpleCRFFrame::set_connectivity(const Connectivity* conn) {
    edges.clear();
    for (int i = 0; i < conn->num_nodes; i++) {
        for(int k = 0; k < conn->num_neighbors[i]; k++) {
            int j = conn->neighbors[i][k];
            edges.at(i).push_back(j);
        }
    }
}

void SimpleCRFFrame::normalize() {
    for (int i = 0; i < num_nodes; i++) {
        float sum = 0;
        for (int cls = 0; cls < num_classes; cls++) {
            sum += q[num_nodes * cls + i];
        }

        for (int cls = 0; cls < num_classes; cls++) {
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

    for (int i = 0; i < num_nodes; i++) {
        int active_cls = classes[i];

        for (int cls = 0; cls < num_classes; cls++) {
            unaries[num_nodes * cls + i] = (cls == active_cls)? active_unary : inactive_unary;
        }
    }
}


void SimpleCRFFrame::set_proba(const float* probas) {
    for (int cls = 0; cls < num_classes; cls++) {
        for (int i = 0; i < num_nodes; i++) {
            unaries[num_nodes * cls + i] = -logf(probas[cls * num_nodes + i]);
        }
    }
}

void SimpleCRFFrame::reset_inferred() {
    for (int cls = 0; cls < num_classes; cls++) {
        for (int i = 0; i < num_nodes; i++) {
            q[num_nodes * cls + i] = expf(-unaries[cls * num_nodes + i]);
        }
    }
}

static inline float pow2(float a) {
    return a * a;
}

float SimpleCRFFrame::calc_temporal_pairwise_energy(int node, const SimpleCRFFrame& other) {
    const Cluster &cluster_1 = clusters[node];
    const Cluster &cluster_2 = other.clusters[node];
    float exponent = -(
        pow2((cluster_1.r - cluster_2.r) / parent.params.temporal_srgb) +
        pow2((cluster_1.g - cluster_2.g) / parent.params.temporal_srgb) +
        pow2((cluster_1.b - cluster_2.b) / parent.params.temporal_srgb)
    ) / 2;
    return parent.params.temporal_w * expf(exponent);
}

float SimpleCRFFrame::calc_spatial_pairwise_energy(int node_i, int node_j) {
    const Cluster &cluster_1 = clusters[node_i];
    const Cluster &cluster_2 = clusters[node_j];
    float exponent = -(
        pow2((cluster_1.r - cluster_2.r) / parent.params.temporal_srgb) +
        pow2((cluster_1.g - cluster_2.g) / parent.params.temporal_srgb) +
        pow2((cluster_1.b - cluster_2.b) / parent.params.temporal_srgb)
    ) / 2;
    return parent.params.spatial_w * expf(exponent);
}

const char* SimpleCRF::inference(int max_iter) {
    int first_time = first_time(), last_time = last_time();

    std::map<int, float *> new_qs; 
    for (int t = first_time; t <= last_time; t++) {
        float *messages = new float[num_nodes * num_classes];
        float *compat_exps = new float[num_nodes * num_classes];

        const SimpleCRFFrame& frame = get_frame(t);
        // Message passing
        for (int cls = 0; cls < num_classes; cls++) {
            for (int i = 0; i < num_nodes; i++) {
                auto & neighbors = frame.connected_nodes(i);
                // pairwise literal
                float message = 0;
                for (auto j : neighbors) {
                    float pair_q = q[num_nodes * cls  + j];
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
        for (int cls = 0; cls < num_classes; cls++) {
            for (int i = 0; i < num_nodes; i++) {
                float gathtered_message_sum = 0;
                for (int other_cls = 0 ; other_cls < num_classes; other_cls++) {
                    if (other_cls == cls) continue; // Potts model
                    gathtered_message_sum += compat_by_class[other_cls] * messages[other_cls * num_nodes + i];
                }
                compat_exps[cls * num_nodes + i] = expf(-(unaries[cls * num_nodes + i] + gathtered_message_sum));
             }
        }

        for (int i = 0; i < num_nodes; i++) {
            float sum = 0;
            for (int cls = 0; cls < num_classes; cls++) {
                sum += compat_exps[num_nodes * cls + i];
            }
            for (int cls = 0; cls < num_classes; cls++) {
                compat_exps[num_nodes * cls + i] /= sum;
            }
        }
        new_qs[t] = compat_exps;
        delete [] messages;
    }

    for (int t = first_time; t <= last_time; t++) {
        SimpleCRFFrame& frame = get_frame(t);
        float *new_q = new_qs[t];
        std::copy(new_q, new_q + (num_nodes * num_classes), frame.q);
        delete [] new_q;
    }

    return nullptr;
}

/*

    // Normalize
    for (int i = 0; i < num_nodes; i++) {
        float norm_exp_sum = 0;
        for (int cls = 0; cls < num_classes; cls++) {
            norm_exp_sum += compat_exps[cls * num_nodes + i];
        }

        for (int cls = 0; cls < num_classes; cls++) {
           compat_exps[cls * num_nodes + i] /= norm_exp_sum;
        }
    }
    return 0;
}
*/
