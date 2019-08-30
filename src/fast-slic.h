#ifndef _FAST_SLIC_H
#define _FAST_SLIC_H
#ifdef __cplusplus

#include <cstring>
#include "fast-slic-common.h"


extern "C" {
#endif
#ifdef __cplusplus
    Connectivity* fast_slic_get_connectivity(int H, int W, int K, const uint16_t *assignment);
    Connectivity* fast_slic_knn_connectivity(int H, int W, int K, const Cluster* clusters, size_t num_neighbors);
    void fast_slic_free_connectivity(Connectivity* conn);
    void fast_slic_get_mask_density(int H, int W, int K, const Cluster* clusters, const uint16_t* assignment, const uint8_t *mask, uint8_t *cluster_densities);
    void fast_slic_cluster_density_to_mask(int H, int W, int K, const Cluster *clusters, const uint16_t* assignment, const uint8_t *cluster_densities, uint8_t *result);
}
#endif

#endif

