#include <vector>
#include <chrono>
#include <cstring>
#include <unordered_set>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include "simd-helper.hpp"
#include "fast-slic-common.h"

typedef std::chrono::high_resolution_clock Clock;

#define CHARBIT 8
#ifdef _MSC_VER
#define __restrict__ __restrict
#endif

#ifndef _ZOrderTuple_DEFINED
#define _ZOrderTuple_DEFINED
struct ZOrderTuple {
    uint32_t score;
    const Cluster* cluster;

    ZOrderTuple(uint32_t score, const Cluster* cluster) : score(score), cluster(cluster) {};
};

static bool operator<(const ZOrderTuple &lhs, const ZOrderTuple &rhs) {
    return lhs.score < rhs.score;
}
#endif


template <typename T>
static inline T my_max(T x, T y) {
    return (x > y) ? x : y;
}


template <typename T>
static inline T my_min(T x, T y) {
    return (x < y) ? x : y;
}


template <typename T>
static T fast_abs(T n)
{
    if (n < 0)
        return -n;
    return n;
}

template <typename T>
static T ceil_int(T numer, T denom) {
    return (numer + denom - 1) / denom;
}

template <typename T>
static T round_int(T numer, T denom) {
    return (numer + (denom / 2)) / denom;
}

class BaseContext {
public:
    int H, W, K;
    int16_t S;
    const char* algorithm;
    float compactness;
    uint8_t quantize_level;
    Cluster* __restrict__ clusters;
    const uint8_t* __restrict__ image = nullptr;
    uint16_t* __restrict__ spatial_dist_patch = nullptr;
    uint16_t* __restrict__ spatial_normalize_cache = nullptr;
    uint32_t* __restrict__ assignment = nullptr;

public:
    virtual ~BaseContext() {
        if (spatial_dist_patch) {
            simd_helper::free_aligned_array(spatial_dist_patch);
        }
        if (spatial_normalize_cache) {
            delete [] spatial_normalize_cache;
        }
    }

    virtual void prepare_spatial() {
        if (spatial_normalize_cache) delete [] spatial_normalize_cache;
        spatial_normalize_cache = new uint16_t[2 * S + 2];
        for (int x = 0; x < 2 * S + 2; x++) {
            // rescale distance [0, 1] to [0, 25.5] (color-scale).
            spatial_normalize_cache[x] = (uint16_t)(compactness * ((float)x / (2 * S) * 25.5f) * (1 << quantize_level));

        }

        const uint16_t patch_height = 2 * S + 1, patch_virtual_width = 2 * S + 1;
        const uint16_t patch_memory_width = simd_helper::align_to_next(patch_virtual_width);

        if (spatial_dist_patch) simd_helper::free_aligned_array(spatial_dist_patch);
        spatial_dist_patch = simd_helper::alloc_aligned_array<uint16_t>(patch_height * patch_memory_width);
        uint16_t row_first_manhattan = 2 * S;
        // first half lines
        for (int i = 0; i < S; i++) {
            uint16_t current_manhattan = row_first_manhattan--;
            // first half columns
            for (int j = 0; j < S; j++) {
                uint16_t val = spatial_normalize_cache[current_manhattan--];
                spatial_dist_patch[i * patch_memory_width + j] = val;
            }
            // half columns next to the first columns
            for (int j = S; j <= 2 * S; j++) {
                uint16_t val = spatial_normalize_cache[current_manhattan++];
                spatial_dist_patch[i * patch_memory_width + j] = val;
            }
        }

        // next half lines
        for (int i = S; i <= 2 * S; i++) {
            uint16_t current_manhattan = row_first_manhattan++;
            // first half columns
            for (int j = 0; j < S; j++) {
                uint16_t val = spatial_normalize_cache[current_manhattan--];
                spatial_dist_patch[i * patch_memory_width + j] = val;
            }
            // half columns next to the first columns
            for (int j = S; j <= 2 * S; j++) {
                uint16_t val = spatial_normalize_cache[current_manhattan++];
                spatial_dist_patch[i * patch_memory_width + j] = val;
            }
        }
    }
};

static uint32_t calc_z_order(uint16_t yPos, uint16_t xPos)
{
    static const uint32_t MASKS[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
    static const uint32_t SHIFTS[] = {1, 2, 4, 8};

    uint32_t x = xPos;  // Interleave lower 16 bits of x and y, so the bits of x
    uint32_t y = yPos;  // are in the even positions and bits from y in the odd;

    x = (x | (x << SHIFTS[3])) & MASKS[3];
    x = (x | (x << SHIFTS[2])) & MASKS[2];
    x = (x | (x << SHIFTS[1])) & MASKS[1];
    x = (x | (x << SHIFTS[0])) & MASKS[0];

    y = (y | (y << SHIFTS[3])) & MASKS[3];
    y = (y | (y << SHIFTS[2])) & MASKS[2];
    y = (y | (y << SHIFTS[1])) & MASKS[1];
    y = (y | (y << SHIFTS[0])) & MASKS[0];

    const uint32_t result = x | (y << 1);
    return result;
}


static uint32_t get_sort_value(int16_t y, int16_t x, int16_t S) {
    return calc_z_order(y, x);
}

static void slic_enforce_connectivity(BaseContext *context) {
    int H = context->H;
    int W = context->W;
    int K = context->K;
    const Cluster* clusters = context->clusters;
    uint32_t* assignment = context->assignment;
    if (K <= 0) return;

    uint8_t *visited = new uint8_t[H * W];
    std::fill_n(visited, H * W, 0);

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int base_index = W * i + j;
            if (assignment[base_index] != 0xFFFF) continue;

            std::vector<int> visited_indices;
            std::vector<int> stack;
            std::unordered_set<int> adj_cluster_indices;
            stack.push_back(base_index);
            while (!stack.empty()) {
                int index = stack.back();
                stack.pop_back();

                if (assignment[index] != 0xFFFF) {
                    adj_cluster_indices.insert(assignment[index]);
                    continue;
                } else if (visited[index]) {
                    continue;
                }
                visited[index] = 1;
                visited_indices.push_back(index);

                int index_j = index % W;
                // up
                if (index > W) {
                    stack.push_back(index - W);
                }

                // down
                if (index + W < H * W) {
                    stack.push_back(index + W);
                }

                // left
                if (index_j > 0) {
                    stack.push_back(index - 1);
                }

                // right
                if (index_j + 1 < W) {
                    stack.push_back(index + 1);
                }
            }

            int target_cluster_index = 0;
            uint32_t max_num_members = 0;
            for (auto it = adj_cluster_indices.begin(); it != adj_cluster_indices.end(); ++it) {
                const Cluster* adj_cluster = &clusters[*it];
                if (max_num_members < adj_cluster->num_members) {
                    target_cluster_index = adj_cluster->number;
                    max_num_members = adj_cluster->num_members;
                }
            }

            for (auto it = visited_indices.begin(); it != visited_indices.end(); ++it) {
                assignment[*it] = target_cluster_index;
            }

        }
    }
    delete [] visited;
}

static void do_fast_slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters) {
    int *gradients = new int[H * W];

    int num_sep = my_max(1, (int)sqrt((double)K));

    int h = H / num_sep;
    int w = W / num_sep;

    // compute gradients
    std::fill_n(gradients, H * W, 1 << 21);
    for (int i = 1; i < H; i += h) {
        for (int j = 1; j < W; j += w) {
            int base_index = i * W + j;
            int img_base_index = 3 * base_index;
            int dx = 
                fast_abs((int)image[img_base_index + 3] - (int)image[img_base_index - 3]) +
                fast_abs((int)image[img_base_index + 4] - (int)image[img_base_index - 2]) +
                fast_abs((int)image[img_base_index + 5] - (int)image[img_base_index - 1]);
            int dy = 
                fast_abs((int)image[img_base_index + 3 * W] - (int)image[img_base_index - 3 * W]) +
                fast_abs((int)image[img_base_index + 3 * W + 1] - (int)image[img_base_index - 3 * W + 1]) +
                fast_abs((int)image[img_base_index + 3 * W + 2] - (int)image[img_base_index - 3 * W + 2]);
            gradients[base_index] = dx + dy;
        }
    }

    int acc_k = 0;
    for (int i = 0; i < H; i += h) {
        for (int j = 0; j < W; j += w) {
            if (acc_k >= K) break;

            int eh = my_min<int>(i + h, H - 1), ew = my_min<int>(j + w, W - 1);
            int center_y = i + h / 2, center_x = j + w / 2;
            int min_gradient = 1 << 21;
            for (int ty = i; ty < eh; ty++) {
                for (int tx = j; tx < ew; tx++) {
                    int base_index = ty * W + tx;
                    if (min_gradient > gradients[base_index]) {
                        center_y = ty;
                        center_x = tx;
                        min_gradient = gradients[base_index];
                    }

                }
            }

            clusters[acc_k].y = center_y;
            clusters[acc_k].x = center_x;


            acc_k++;
        }
    }

    while (acc_k < K) {
        clusters[acc_k].y = H / 2;
        clusters[acc_k].x = W / 2;
        acc_k++;
    }

    delete [] gradients;


    for (int k = 0; k < K; k++) {
        int base_index = W * clusters[k].y + clusters[k].x;
        int img_base_index = 3 * base_index;
        clusters[k].r = image[img_base_index];
        clusters[k].g = image[img_base_index + 1];
        clusters[k].b = image[img_base_index + 2];
        clusters[k].number = k;
        clusters[k].num_members = 0;
    }
}

