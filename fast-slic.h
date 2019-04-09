#ifndef _FAST_SLIC_H
#define _FAST_SLIC_H
#include <stdint.h>

typedef uint16_t cluster_no_t;

typedef struct Cluster {
    // 7 bytes
    uint16_t y;
    uint16_t x;
    uint8_t r;
    uint8_t g;
    uint8_t b;
    // 1 byte dummy data
    uint8_t reserved[1];

    cluster_no_t number; // 2 bytes
    uint8_t is_active;
    uint32_t num_members;

} Cluster;

#ifdef __cplusplus
extern "C" {
#endif
    void slic_assign(int H, int W, int K, uint8_t compactness_shift, uint8_t quantize_level, const uint8_t* image, const Cluster* clusters, uint32_t* assignment);
    void slic_update_clusters(int H, int W, int K, const uint8_t* image, Cluster* clusters, const uint32_t* assignment);
    void slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters);
    void do_slic(int H, int W, int K, uint8_t compactness_shift, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint32_t* assignment);
#ifdef __cplusplus
}
#endif

#endif

