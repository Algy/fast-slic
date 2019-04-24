#include <vector>
#include <chrono>
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <climits>
#include <list>
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


class ConnectedComponent {
public:
    uint32_t cluster_no;
    std::vector< std::shared_ptr<std::vector<int>> > indices;
    int num_indices;
    const Cluster* max_adjacent_cluster;
public:
    ConnectedComponent(uint32_t cluster_no) : cluster_no(cluster_no), num_indices(0), max_adjacent_cluster(nullptr) {
        indices.push_back(std::shared_ptr< std::vector<int> > { new std::vector<int>() });
    };

    void merge(ConnectedComponent &&other) {
        indices.insert(indices.end(), other.indices.begin(), other.indices.end());
        num_indices += other.num_indices;

        if (other.max_adjacent_cluster != nullptr &&
                (max_adjacent_cluster == nullptr || max_adjacent_cluster->num_members < other.max_adjacent_cluster->num_members)) {
            max_adjacent_cluster = other.max_adjacent_cluster;
        }

        other.num_indices = 0;
        other.indices.clear();
        other.max_adjacent_cluster = nullptr;
    }

    bool empty() {
        return num_indices <= 0;
    }

    inline void add(int index) {
        indices.back()->push_back(index);
        num_indices++;
    }

    inline void inform_adjacent_cluster(const Cluster* cluster) {
        if (max_adjacent_cluster == nullptr || max_adjacent_cluster->num_members < cluster->num_members) {
            max_adjacent_cluster = cluster;
        }
    }
};

class ConnectedComponentSet {
private:
    std::vector< ConnectedComponent > components;
    std::vector<int> parents;
    std::vector<int> ranks;
public:
    inline int add_component(uint32_t cluster_no) {
        int id = (int)parents.size();
        parents.push_back(id);
        ranks.push_back(1);
        components.push_back(ConnectedComponent(cluster_no));
        return id;
    }

    inline ConnectedComponent& find(int id) {
        int parent = parents[id];
        if (parent == id) return components[id]; // fast path
        int root_id = find_root_with_compression(parent);
        parents[id] = root_id;
        return components[root_id];
    }

    // Union by rank
    inline int merge(int lhs_id, int rhs_id) {
        int lhs_root_id = find_root_with_compression(lhs_id), rhs_root_id = find_root_with_compression(rhs_id);
        if (lhs_root_id == rhs_root_id) return lhs_id;

        ConnectedComponent& lhs = components[lhs_root_id], rhs = components[rhs_root_id];

        if (ranks[lhs_root_id] < ranks[rhs_root_id]) {
            parents[lhs_root_id] = rhs_root_id;
        } else if (ranks[lhs_root_id] > ranks[rhs_root_id]) {
            parents[rhs_root_id] = lhs_root_id;
        } else {
            parents[rhs_root_id] = lhs_root_id;
            ranks[lhs_root_id]++;
        }
        lhs.merge(std::move(rhs));
        return lhs_root_id;
    }

    std::vector< ConnectedComponent * > connected_components() {
        std::vector< ConnectedComponent *> results;
        for (auto &cc : components) {
            if (cc.empty()) continue;
            results.push_back(&cc);
        }
        return results;
    }

private:
    // Path compression
    int find_root_with_compression(int id) {
        if (id == parents[id]) return id;
        int root = find_root_with_compression(parents[id]);
        parents[id] = root;
        return root;
    }
};

static void fast_enforce_connectivity(BaseContext* context) {
    int H = context->H;
    int W = context->W;
    int K = context->K;
    int S = context->S;
    if (K <= 0 || H <= 0 || W <= 0) return;

    const Cluster* clusters = context->clusters;
    uint32_t* assignment = context->assignment;

    ConnectedComponentSet cc_set;

    std::unique_ptr< std::vector<int> > prev_cc_row { new std::vector<int>() };

    // (0, 0)
    int left_id = cc_set.add_component(assignment[0]);
    ConnectedComponent* left_cc = &cc_set.find(left_id);
    prev_cc_row->push_back(left_id);

    for (int j = 1; j < W; j++) {
        uint32_t cluster_no = assignment[j];
        if (left_cc->cluster_no == cluster_no) {
            left_cc->add(j);
        } else {
            if (cluster_no != 0xFFFF) {
                left_cc->inform_adjacent_cluster(&clusters[cluster_no]);
            }
            int new_id = cc_set.add_component(cluster_no);
            left_cc = &cc_set.find(new_id);
            left_id = new_id;
        }
        prev_cc_row->push_back(left_id);
    }

    for (int i = 1; i < H; i++) {
        std::unique_ptr< std::vector<int> > curr_cc_row { new std::vector<int>() };

        int left_id;
        ConnectedComponent* left_cc;
        {
            uint32_t cluster_no = assignment[i * W];
            int up_id = (*prev_cc_row)[0];
            ConnectedComponent* up_cc = &cc_set.find(up_id);
            if (up_cc->cluster_no == cluster_no) {
                left_id = up_id;
                left_cc = up_cc;
            } else {
                int new_id = cc_set.add_component(cluster_no);
                up_cc->inform_adjacent_cluster(&clusters[cluster_no]);
                left_id = new_id;
                left_cc = &cc_set.find(new_id);
            }
        }
        curr_cc_row->push_back(left_id);

        for (int j = 1; j < W; j++) {
            int index = i * W + j;
            uint32_t cluster_no = assignment[index];
            int up_id = (*prev_cc_row)[j];
            ConnectedComponent* up_cc = &cc_set.find(up_id);

            bool left_mergable = left_cc->cluster_no == cluster_no;
            bool up_mergable = up_cc->cluster_no == cluster_no;

            if (left_mergable && up_mergable) {
                int merged_id = cc_set.merge(left_id, up_id);
                left_id = merged_id;
                left_cc = &cc_set.find(merged_id);
            } else if (left_mergable) {
                left_cc->add(index);
                if (cluster_no != 0xFFFF)
                    up_cc->inform_adjacent_cluster(&clusters[cluster_no]);
            } else if (up_mergable) {
                up_cc->add(index);
                if (cluster_no != 0xFFFF)
                    left_cc->inform_adjacent_cluster(&clusters[cluster_no]);
            } else {
                if (cluster_no != 0xFFFF) {
                    left_cc->inform_adjacent_cluster(&clusters[cluster_no]);
                    up_cc->inform_adjacent_cluster(&clusters[cluster_no]);
                }
                int new_id = cc_set.add_component(cluster_no);
                left_id = new_id;
                left_cc = &cc_set.find(new_id);
            }
            curr_cc_row->push_back(left_id);
        }
        prev_cc_row = std::move(curr_cc_row);
    }

    int thres = S * S / 2;
    for (ConnectedComponent *cc : cc_set.connected_components()) {
        if (cc->cluster_no == 0xFFFF || cc->num_indices < thres) {
            uint32_t new_cluster_no = (cc->max_adjacent_cluster != nullptr)? cc->max_adjacent_cluster->number : 0;
            for (const std::shared_ptr<std::vector<int>> &vector_ptr : cc->indices) {
                for (int index : *vector_ptr) {
                    assignment[index] = new_cluster_no;
                }
            }
        }
    }
}

static void slic_enforce_connectivity(BaseContext *context) {
    fast_enforce_connectivity(context);
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

