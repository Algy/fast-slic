#ifndef _FAST_SLIC_AVX2_H
#define _FAST_SLIC_AVX2_H
#include <stdint.h>

#include "fast-slic-common.h"

#ifdef __cplusplus
extern "C" {
#endif
    void fast_slic_initialize_clusters_avx2(int H, int W, int K, const uint8_t* image, Cluster *clusters);
    void fast_slic_iterate_avx2(int H, int W, int K, float compactness, float min_size_factor, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint16_t* assignment);
    int fast_slic_supports_avx2();
#ifdef __cplusplus
}
#endif

#endif

