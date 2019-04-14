#ifndef _FAST_SLIC_H
#define _FAST_SLIC_H
#ifdef __cplusplus

#include "fast-slic-common.h"


extern "C" {
#endif
    void fast_slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters);
    void fast_slic_iterate(int H, int W, int K, uint8_t compactness, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint32_t* assignment);
#ifdef __cplusplus
}
#endif

#endif

