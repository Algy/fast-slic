#ifndef _FAST_SLIC_COMMON_H
#define _FAST_SLIC_COMMON_H

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

typedef struct Connectivity {
    int num_nodes;
    int *num_neighbors;
    uint32_t **neighbors;
} Connectivity;

#endif
