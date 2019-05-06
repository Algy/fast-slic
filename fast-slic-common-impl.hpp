#include <vector>
#include <chrono>
#include <cassert>
#include <cstring>
#include <string>
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
#include <atomic>
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
static inline T clamp(T value, T lo, T hi) {
    return (value < lo) ? lo : ((value > hi)? hi : value);
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
    float min_size_factor = 0.1;
    uint8_t quantize_level;
    Cluster* __restrict__ clusters;
    const uint8_t* __restrict__ image = nullptr;
    uint16_t* __restrict__ spatial_dist_patch = nullptr;
    uint16_t* __restrict__ spatial_normalize_cache = nullptr;
    uint16_t* __restrict__ assignment = nullptr;

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


class FlatCCSet {
public:
    int* component_assignment;

    int num_components;
    std::vector<int> num_component_members;
    std::vector<uint16_t> component_cluster_nos;
    std::vector<const Cluster *>max_component_adj_clusters;
public:
    FlatCCSet(int image_size) : component_assignment(new int[image_size]) {
        std::fill_n(component_assignment, image_size, -1);
    };
    FlatCCSet(const FlatCCSet& other) = delete;
    FlatCCSet& operator=(const FlatCCSet& other) = delete;
    ~FlatCCSet() { delete [] component_assignment; }
};

class ConnectedComponentSet {
public:
    int size;
    std::vector<int> parents;
private:
    std::vector<const Cluster *> max_adj_clusters;
public:
    ConnectedComponentSet() : size(0) {};
    ConnectedComponentSet(int size) : size(size), parents(size), max_adj_clusters(size, nullptr) {
        for (int i = 0; i < size; i++) {
            parents[i] = i;
        }
    }

    void clear_cluster_info() {
        std::fill(max_adj_clusters.begin(), max_adj_clusters.end(), nullptr);
    }

private:
    inline int find_root(int node) {
        int parent = parents[node];
        while (parent < node) {
            node = parent;
            parent = parents[parent];
        }
        return node;
    }

    inline void set_root(int root, int node) {
        for (int i = node; root < i; ) {
            int parent = parents[i];
            parents[i] = root;
            i = parent;
        }
    }
public:
    inline int add_new_component() {
        int c = size++;
        parents.push_back(c);
        max_adj_clusters.push_back(nullptr);
        return c;
    }

    inline int find(int node) {
        int root = find_root(node);
        set_root(root, node);
        return root;
    }

    inline int merge_roots(int root_i, int root_j) {
        int root = root_i;
        if (root_j != root) {
            if (root > root_j) std::swap(root, root_j);
            if (max_adj_clusters[root] != nullptr && max_adj_clusters[root_j] != nullptr && max_adj_clusters[root]->num_members < max_adj_clusters[root_j]->num_members) {
                max_adj_clusters[root] = max_adj_clusters[root_j];
            }
        }
        return root;
    }

    inline int merge(int node_i, int node_j) {
        int root = find_root(node_i);
        int root_j = find_root(node_j);
        root = merge_roots(root, root_j);
        set_root(root, node_j);
        set_root(root, node_i);
        return root;
    }

    inline void add_single(int node_i, int single_j) {
        parents[single_j] = parents[node_i];
    }

    void inform_adjacent_cluster(int node, const Cluster *cluster) {
        int root = find(node);
        if (max_adj_clusters[root] == nullptr || max_adj_clusters[root]->num_members < cluster->num_members) {
            max_adj_clusters[root] = cluster;
        }
    }

    inline std::shared_ptr<FlatCCSet> flatten(const uint16_t *assignment) {
        int size = (int)parents.size();
        std::shared_ptr<FlatCCSet> result { new FlatCCSet(size) };
        std::atomic<int> component_counter { 0 };
        #pragma omp parallel
        {
            // rename leading nodes
            #pragma omp for
            for (int i = 0; i < size; i++) {
                if (parents[i] == i) {
                    result->component_assignment[i] = component_counter++;
                }
            }

            #pragma omp single
            {
                result->num_components = component_counter.load();
                result->num_component_members.resize(result->num_components, 0);
                result->component_cluster_nos.resize(result->num_components);
                result->max_component_adj_clusters.resize(result->num_components);
            }

            std::vector<int> local_num_component_members;
            local_num_component_members.resize(result->num_components, 0);
            #pragma omp for
            for (int i = 0; i < size; i++) {
                int parent = parents[i];
                if (parent < i) {
                    int component_no = result->component_assignment[parent];
                    while (component_no == -1) {
                        parent = parents[parent];
                        component_no = result->component_assignment[parent];
                    }
                    result->component_assignment[i] = component_no;
                    local_num_component_members[component_no]++;
                } else {
                    int component_no = result->component_assignment[i];
                    result->component_cluster_nos[component_no] = assignment[i];
                    result->max_component_adj_clusters[component_no] = max_adj_clusters[i];
                    local_num_component_members[component_no]++;
                }
            }

            #pragma omp critical
            for (int i = 0; i < result->num_components; i++) {
                result->num_component_members[i] += local_num_component_members[i];
            }
        }
        return result;
    }
};

static void build_cc_set(ConnectedComponentSet &cc_set, const Cluster* clusters, int H, int W, uint16_t *assignment) {
    std::vector<int> seam_ys;
    #pragma omp parallel
    {
        bool is_first = true;
        int seam = 0;
        #pragma omp for
        for (int i = 0; i < H; i++) {
            if (is_first) {
                is_first = false;
                seam = i;
                uint16_t left_cluster_no = assignment[i * W];
                for (int j = 1; j < W; j++) {
                    int index = i * W + j;
                    uint16_t cluster_no = assignment[index];
                    if (left_cluster_no == cluster_no) {
                        cc_set.merge(index - 1, index);
                    } else if (cluster_no != 0xFFFF) {
                        cc_set.inform_adjacent_cluster(index - 1, &clusters[cluster_no]);
                    }
                    left_cluster_no = cluster_no;
                }
                continue;
            }

            uint16_t left_cluster_no;
            {
                int index = i * W;
                int up_index = index - W;
                uint16_t cluster_no = assignment[index];
                if (assignment[up_index] == cluster_no) {
                    cc_set.merge(up_index, index);
                } else if (cluster_no != 0xFFFF) {
                    cc_set.inform_adjacent_cluster(up_index, &clusters[cluster_no]);
                }
                left_cluster_no = cluster_no;
            }
            for (int j = 1; j < W; j++) {
                int index = i * W + j;
                uint16_t cluster_no = assignment[index];
                int left_index = index - 1, up_index = index - W;

                if (left_cluster_no == cluster_no) {
                    if (assignment[up_index] == cluster_no) {
                        if (cc_set.parents[left_index] == cc_set.parents[up_index]) {
                            cc_set.add_single(left_index, index);
                        } else {
                            cc_set.add_single(left_index, index);
                            cc_set.merge(left_index, up_index);
                        }
                    } else {
                        cc_set.add_single(left_index, index);
                        if (cluster_no != 0xFFFF) {
                            cc_set.inform_adjacent_cluster(up_index, &clusters[cluster_no]);
                        }
                    }
                } else {
                    if (cluster_no != 0xFFFF) {
                        cc_set.inform_adjacent_cluster(left_index, &clusters[cluster_no]);
                        if (assignment[up_index] == cluster_no) {
                            cc_set.add_single(up_index, index);
                        } else {
                            cc_set.inform_adjacent_cluster(up_index, &clusters[cluster_no]);
                        }
                    } else {
                        if (assignment[up_index] == cluster_no) {
                            cc_set.add_single(up_index, index);
                        }
                    }
                }

                left_cluster_no = cluster_no;
            }
        }

        #pragma omp critical
        seam_ys.push_back(seam);
    }

    for (int i : seam_ys) {
        if (i <= 0) continue;
        for (int j = 0; j < W; j++) {
            int index = i * W + j;
            int up_index = index - W;
            uint16_t cluster_no = assignment[index];
            if (assignment[up_index] == cluster_no) {
                cc_set.merge(index, up_index);
            } else if (cluster_no != 0xFFFF) {
                cc_set.inform_adjacent_cluster(up_index, &clusters[cluster_no]);
            }
        }
    }
}

static void merge_cc_set(ConnectedComponentSet &cc_set, const Cluster* clusters, int H, int W, uint16_t *assignment) {
    std::vector<int> seam_ys;
    #pragma omp parallel
    {
        bool is_first = true;
        int seam = 0;
        #pragma omp for
        for (int i = 0; i < H; i++) {
            if (is_first) {
                is_first = false;
                seam = i;
                uint16_t left_cluster_no = assignment[i * W];
                for (int j = 1; j < W; j++) {
                    int index = i * W + j;
                    uint16_t cluster_no = assignment[index];
                    if (left_cluster_no == 0xFFFF) {
                        if (cluster_no == 0xFFFF) {
                            cc_set.merge(index - 1, index);
                        } else {
                            cc_set.inform_adjacent_cluster(index - 1, &clusters[cluster_no]);
                        }
                    }
                    left_cluster_no = cluster_no;
                }
                continue;
            }

            uint16_t left_cluster_no;
            {
                int index = i * W;
                int up_index = index - W;
                uint16_t cluster_no = assignment[index];
                uint16_t up_cluster_no = assignment[up_index];
                if (up_cluster_no == 0xFFFF){
                    if (cluster_no == 0xFFFF) {
                        cc_set.merge(up_index, index);
                    } else {
                        cc_set.inform_adjacent_cluster(up_index, &clusters[cluster_no]);
                    }
                }
                left_cluster_no = cluster_no;
            }
            for (int j = 1; j < W; j++) {
                int index = i * W + j;
                uint16_t cluster_no = assignment[index];
                int left_index = index - 1, up_index = index - W;
                uint16_t up_cluster_no = assignment[up_index];

                if (cluster_no == 0xFFFF) {
                    if (left_cluster_no == 0xFFFF && cc_set.parents[left_index] != cc_set.parents[index]) {
                        cc_set.merge(left_index, index);
                    }
                    if (up_cluster_no == 0xFFFF && cc_set.parents[up_index] != cc_set.parents[index]) {
                        cc_set.merge(up_index, index);
                    }
                } else {
                    if (left_cluster_no == 0xFFFF) {
                        cc_set.inform_adjacent_cluster(left_index, &clusters[cluster_no]);
                    }
                    if (up_cluster_no == 0xFFFF) {
                        cc_set.inform_adjacent_cluster(up_index, &clusters[cluster_no]);
                    }
                }

                left_cluster_no = cluster_no;
            }
        }

        #pragma omp critical
        seam_ys.push_back(seam);
    }

    for (int i : seam_ys) {
        if (i <= 0) continue;
        for (int j = 0; j < W; j++) {
            int index = i * W + j;
            int up_index = index - W;
            uint16_t cluster_no = assignment[index];
            if (assignment[up_index] == 0xFFFF) {
                if (cluster_no == 0xFFFF) {
                    cc_set.merge(up_index, index);
                } else {
                    cc_set.inform_adjacent_cluster(up_index, &clusters[cluster_no]);
                }
            }
        }
    }
}

static void fast_remove_blob(BaseContext* context) {
    int H = context->H;
    int W = context->W;
    int K = context->K;
    int S = context->S;
    if (K <= 0 || H <= 0 || W <= 0) return;

    const Cluster* clusters = context->clusters;
    uint16_t* assignment = context->assignment;

    ConnectedComponentSet cc_set(H * W);

    // auto t1 = Clock::now();
    build_cc_set(cc_set, clusters, H, W, assignment);
    // auto t21 = Clock::now();
    std::shared_ptr<FlatCCSet> flat_cc = cc_set.flatten(assignment);
    int thres = (int)round((double)(S * S) * (double)context->min_size_factor);

    // auto t2 = Clock::now();

    // auto t3 = Clock::now();
    #pragma omp parallel for
    for (int i = 0; i < H * W; i++) {
        if (flat_cc->num_component_members[flat_cc->component_assignment[i]] < thres) {
            assignment[i] = 0xFFFF;
        }
    }

    // auto t4 = Clock::now();
    cc_set.clear_cluster_info();
    merge_cc_set(cc_set, clusters, H, W, assignment);
    std::shared_ptr<FlatCCSet> flat_blank_cc = cc_set.flatten(assignment);
    // auto t5 = Clock::now();

    std::vector<uint16_t> sub_clsuter_nos(flat_blank_cc->num_components, 0xFFFF);

    #pragma omp parallel
    {
        #pragma omp for
        for (int k = 0; k < flat_blank_cc->num_components; k++) {
            if (flat_blank_cc->component_cluster_nos[k] != 0xFFFF) continue;
            const Cluster *cluster = flat_blank_cc->max_component_adj_clusters[k];
            sub_clsuter_nos[k] = (cluster != nullptr)? cluster->number: 0;
        }

        #pragma omp for
        for (int i = 0; i < H * W; i++) {
            uint16_t sub_cluster_no = sub_clsuter_nos[flat_blank_cc->component_assignment[i]];
            if (sub_cluster_no != 0xFFFF) {
                assignment[i] = sub_cluster_no;
            }
        }
    }
    // auto t6 = Clock::now();

    // std::cerr << "merge: " << std::chrono::duration_cast<std::chrono::microseconds>(t21 - t1).count() << " us" << std::endl;
    // std::cerr << "flatten: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t21).count() << " us" << std::endl;
    // std::cerr << "set to 0xFFFF : " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << " us" << std::endl;
    // std::cerr << "flatten for blank: " << std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count() << " us" << std::endl;
    // std::cerr << "substitute: " << std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count() << " us" << std::endl;
}

static void do_slic_enforce_connectivity_dfs(BaseContext *context) {
    int H = context->H;
    int W = context->W;
    int K = context->K;
    const Cluster* clusters = context->clusters;
    uint16_t* assignment = context->assignment;
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


static void slic_enforce_connectivity(BaseContext *context) {
    if (context->min_size_factor <= 0) {
        do_slic_enforce_connectivity_dfs(context);
    } else {
        fast_remove_blob(context);
    }
}

static void do_fast_slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters) {
    if (H <= 0 || W <= 0 || K <= 0) return;
    int n_y = (int)sqrt((double)K);

    std::vector<int> n_xs(n_y, K / n_y);

    int remainder = K % n_y;
    int row = 0;
    while (remainder-- > 0) {
        n_xs[row]++;
        row += 2;
        if (row >= n_y) {
            row = 1 % n_y;
        }
    }

    int h = ceil_int(H, n_y);
    int acc_k = 0;
    for (int i = 0; i < H; i += h) {
        int w = ceil_int(W, n_xs[my_min<int>(i / h, n_y - 1)]);
        for (int j = 0; j < W; j += w) {
            if (acc_k >= K) {
                break;
            }
            int center_y = i + h / 2, center_x = j + w / 2;
            clusters[acc_k].y = clamp(center_y, 0, H - 1);
            clusters[acc_k].x = clamp(center_x, 0, W - 1);

            acc_k++;
        }
    }

    while (acc_k < K) {
        clusters[acc_k].y = H / 2;
        clusters[acc_k].x = W / 2;
        acc_k++;
    }

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



