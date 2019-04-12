#include <cmath>
#include "simple-crf.hpp"

static inline float logsumexpf_array(float* array, int begin, int end, int stride = 1) {
    int length = (end - begin) / stride;
    if (length == 0)
        return;
    float buffer[length];
    float max_value;
    buffer[0] = max_value = array[begin];
    for (int i = 1; i < length; i++) {
        float t;
        buffer[i] = t = array[begin + i * stride];
        if (max_value < t) {
            max_value = t;
        }
    }

    for (int i = 0; i < length; i++) {
        buffer[i] = expf(buffer[i] - max_value);
    }

    float sum = 0;
    for (int i = 0; i < length; i++) {
        sum += buffer[i];
    }
    return max_value + logf(sum);
}

void CRFTimeFrame::normalize() {
    for (int i = 0; i < num_nodes; i++) {
        float log_denom = logsumexpf_array(log_q, i, num_classes * num_nodes, num_nodes);
        for (int cls = 0; cls < num_classes; cls++) {
            log_q[num_nodes * cls + i] -= log_denom;
        }
    }
}


void CRFTimeFrame::set_connectivity(CRFTimeFrame& frame, const SimpleCRFConnectivity *conn ) {
    pairwise_edges.clear();
    for (int i = 0; i < connectivity->num_nodes; i++) {
        std::vector<CRFPairwise> pairwise_list;
        for (int i = 0; i < connectivity->num_nodes; i++) {
            const int *neightbors_of_i = connectivity->neighbors[i];
            for (int j = 0; j < connectivity->num_neighbors[i]; j++) {
                int neighbor_of_i = neightbors_of_i[j];
                pairwise_list.push_back(CRFPairwise(i, j))
            }
        }
        edges.push_back(pairwise_list)
    }
    pairwise_configured = true;
}
        CRFPixel pixel;
        float dr = (int)node_rgbs[3 * i];
        float dr = (int)node_rgbs[3 * i] - (int)node_rgbs[3 * neighbor_of_i];
        float db = (int)node_rgbs[3 * i + 2] - (int)node_rgbs[3 * neighbor_of_i + 2];
    }
