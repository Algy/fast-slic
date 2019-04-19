#ifndef _FAST_SLIC_H
#define _FAST_SLIC_H
#ifdef __cplusplus

#include "fast-slic-common.h"


extern "C" {
#endif
    void fast_slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters);
    void fast_slic_iterate(int H, int W, int K, uint8_t compactness, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint32_t* assignment);
#ifdef __cplusplus
    Connectivity* fast_slic_get_connectivity(int H, int W, int K, const uint32_t *assignment);
    Connectivity* fast_slic_knn_connectivity(int H, int W, int K, const Cluster* clusters, size_t num_neighbors);
    void fast_slic_free_connectivity(Connectivity* conn);
    void fast_slic_get_mask_density(int H, int W, int K, const Cluster* clusters, const uint32_t* assignment, const uint8_t *mask, uint8_t *cluster_densities);
    void fast_slic_cluster_density_to_mask(int H, int W, int K, const Cluster *clusters, const uint32_t* assignment, const uint8_t *cluster_densities, uint8_t *result);
}
#endif

#endif

