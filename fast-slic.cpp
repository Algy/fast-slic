#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <cstring>
#include <cassert>

#include <immintrin.h>
#include "fast-slic.h"
#include "simd-helper.hpp"

#define CHARBIT 8

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

struct ClusterPixel {
    uint16_t cluster_nos[9];
    int8_t last_index;
};



struct Context {
    const uint8_t* __restrict__ image;
    uint8_t* __restrict__ aligned_quad_image; // copied image
    uint16_t quad_image_memory_width;

    const char* algorithm;
    int H;
    int W;
    int K;
    int16_t S;
    uint8_t compactness_shift;
    uint8_t quantize_level;
    Cluster* __restrict__ clusters;
    uint32_t* __restrict__ assignment;
    uint32_t* __restrict__ aligned_assignment;
    uint16_t* __restrict__ spatial_normalize_cache; // (x) -> (uint16_t)(((uint32_t)x << quantize_level) * M / S / 2 * 3) 
    ClusterPixel* __restrict__ cluster_boxes;
};

static void init_context(Context *context) {
    memset(context, 0, sizeof(Context));
}

static void free_context(Context *context) {
    if (context->cluster_boxes)
        delete [] context->cluster_boxes;
    if (context->spatial_normalize_cache)
        delete [] context->spatial_normalize_cache;
    if (context->aligned_quad_image) {
        simd_helper::free_aligned_array(context->aligned_quad_image);
    }
}

uint32_t calc_z_order(uint16_t yPos, uint16_t xPos)
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


static uint64_t get_sort_value(int16_t y, int16_t x, int16_t S) {
    // return ((uint64_t)(y / (2 * S)) << 48) + ((uint64_t)(x / (2 * S)) << 32) + (uint32_t)calc_z_order(y, x);
    return calc_z_order(y, x);
    // return y + x;
}


struct sort_cmp {
    int16_t S;
    sort_cmp(int16_t S) : S(S) {};
    inline bool operator() (const Cluster * lhs, const Cluster * rhs) {
        return get_sort_value(lhs->y, lhs->x, S) < get_sort_value(rhs->y, rhs->x, S);
    }
};


#include <string>
#include <chrono>
#include <fstream>
#include <memory>
#include <iostream>
typedef std::chrono::high_resolution_clock Clock;

static void slic_assign_cluster_oriented(Context *context) {
    auto H = context->H;
    auto W = context->W;
    auto K = context->K;
    auto compactness_shift = context->compactness_shift;
    auto clusters = context->clusters;
    auto assignment = context->assignment;
    auto quantize_level = context->quantize_level;
    const int16_t S = context->S;

    uint8_t* __restrict__ aligned_quad_image = context->aligned_quad_image;

    auto quad_image_memory_width = context->quad_image_memory_width;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            assignment[i * W + j] = 0xFFFFFFFF;
        }
    }

    /*
     * Calculate Spatial normalizer
     */

    if (!context->spatial_normalize_cache) {
        context->spatial_normalize_cache = new uint16_t[2 * S + 2];
        for (int x = 0; x < 2 * S + 2; x++) {
            context->spatial_normalize_cache[x] = (uint16_t)(((uint32_t)x * compactness_shift << quantize_level) / S / 2 * 3);
        }
    }

    const uint16_t* __restrict__ spatial_normalize_cache = context->spatial_normalize_cache;

    const uint16_t patch_height = 2 * S + 1, patch_virtual_width = 2 * S + 1;
    const uint16_t patch_memory_width = simd_helper::align_to_next(patch_virtual_width);
    uint16_t* spatial_dist_patch = simd_helper::alloc_aligned_array<uint16_t>(patch_height * patch_memory_width);

    {
        uint16_t row_first_manhattan = 2 * S;
        // first half lines
        for (int i = 0; i < S; i++) {
            uint16_t current_manhattan = row_first_manhattan--;
            // first half columns
            for (int j = 0; j < S; j++) {
                spatial_dist_patch[i * patch_memory_width + j] = spatial_normalize_cache[current_manhattan--];
            }
            // half columns next to the first columns
            for (int j = S; j <= 2 * S; j++) {
                spatial_dist_patch[i * patch_memory_width + j] = spatial_normalize_cache[current_manhattan++];
            }
        }

        // next half lines
        for (int i = S; i <= 2 * S; i++) {
            uint16_t current_manhattan = row_first_manhattan++;
            // first half columns
            for (int j = 0; j < S; j++) {
                spatial_dist_patch[i * patch_memory_width + j] = spatial_normalize_cache[current_manhattan--];
            }
            // half columns next to the first columns
            for (int j = S; j <= 2 * S; j++) {
                spatial_dist_patch[i * patch_memory_width + j] = spatial_normalize_cache[current_manhattan++];
            }
        }
    }

    // Sorting clusters by morton order seems to help for distributing clusters evenly for multiple cores
    std::vector<const Cluster *> cluster_sorted_ptrs;
    for (int k = 0; k < K; k++) { cluster_sorted_ptrs.push_back(&clusters[k]); }
    std::stable_sort(cluster_sorted_ptrs.begin(), cluster_sorted_ptrs.end(), sort_cmp(S));

    auto t1 = Clock::now();
 
    #pragma omp parallel for schedule(static)
    for (int cluster_sorted_idx = 0; cluster_sorted_idx < K; cluster_sorted_idx++) {
        const Cluster *cluster = cluster_sorted_ptrs[cluster_sorted_idx];
        cluster_no_t cluster_number = cluster->number;
        const int16_t cluster_y = my_min<int16_t>(my_max<int16_t>(cluster->y, S), H - S - 1), cluster_x = my_min<int16_t>(my_max<int16_t>(cluster->x, S), W - S - 1);
        const int16_t y_lo = cluster_y - S, x_lo = cluster_x - S;

        // Note: x86-64 is little-endian arch. ABGR order is correct.
        const uint32_t cluster_color_quad = ((uint32_t)cluster->b << 16) + ((uint32_t)cluster->g << 8) + ((uint32_t)cluster->r);
        __m256i cluster_color_vec = _mm256_set1_epi32(cluster_color_quad);
        // 16 elements uint16_t (among there elements are the first 8 elements used)
        __m256i cluster_number_vec = _mm256_set1_epi16(cluster_number);

        for (int16_t i = 0; i < patch_height; i++) {
            uint8_t *img_quad_base_row = aligned_quad_image + quad_image_memory_width * (y_lo + i);
            uint16_t *spatial_dist_patch_base_row = spatial_dist_patch + patch_memory_width * i;
            uint32_t* assignment_base_row = assignment + (i + y_lo) * W + x_lo;

            // 32(alignment) / 4(rgba quad) = stride 8 
            for (int j = 0; j < patch_virtual_width; j += 8) {
                // Image rows are aligned
                uint8_t* img_quad_row = img_quad_base_row + 4 * j;
                img_quad_row = (uint8_t *)HINT_ALIGNED_AS(img_quad_row, 32);

                // Spatial distance patch is aligned
                uint16_t* spatial_dist_patch_row = spatial_dist_patch_base_row + j;
                spatial_dist_patch_row = (uint16_t *)HINT_ALIGNED_AS(spatial_dist_patch_row, 32);

                // unaligned
                // TODO: align assignment array
                uint32_t* assignment_row = assignment_base_row + j;

                // image_segment: 32 elements of uint8_t 
                // color_dist_vec, spatial_dist_vec, dist_vec: 8 elements of uint16_t (trailing 8 empty 16-bit cells left)
                // assignment_value_vec: 
                //   8 elements of uint32_t
                //   [high 16-bit: distance value] + [low 16-bit: cluster_number]
                __m256i image_segment = _mm256_load_si256((__m256i*)img_quad_row);
                __m256i color_dist_vec = _mm256_mpsadbw_epu8(image_segment, cluster_color_vec, 0);
                __m256i spatial_dist_vec = _mm256_load_si256((__m256i*)spatial_dist_patch_row);
                __m256i dist_vec = _mm256_adds_epu16(color_dist_vec, spatial_dist_vec);
                __m256i assignment_value_vec = _mm256_unpacklo_epi16(cluster_number_vec, dist_vec);
                // min-assignment
                __m256i min_assignment_vec = _mm256_max_epu32(_mm256_loadu_si256((__m256i*)assignment_row), assignment_value_vec);
                _mm256_storeu_si256((__m256i*)assignment_row, min_assignment_vec);
            }
        }
    }
    auto t2 = Clock::now();

    simd_helper::free_aligned_array(spatial_dist_patch);

    // Clean up: Drop distance part in assignment and let only cluster numbers remain
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            assignment[i * W + j] &= 0x0000FFFF; // drop the leading 2 bytes
        }
    }

    std::cerr << "Tightloop: " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";

}

static void slic_assign(Context *context) {
    if (!strcmp(context->algorithm, "cluster_oriented")) {
        slic_assign_cluster_oriented(context);
    }
}

static void slic_update_clusters(Context *context) {
    auto H = context->H;
    auto W = context->W;
    auto K = context->K;
    auto image = context->image;
    auto clusters = context->clusters;
    auto assignment = context->assignment;

    int num_cluster_members[K];
    int cluster_acc_vec[K][5]; // sum of [y, x, r, g, b] in cluster

    std::fill_n(num_cluster_members, K, 0);
    std::fill_n((int *)cluster_acc_vec, K * 5, 0);

    #pragma omp parallel
    {
        int local_acc_vec[K][5]; // sum of [y, x, r, g, b] in cluster
        int local_num_cluster_members[K];
        std::fill_n(local_num_cluster_members, K, 0);
        std::fill_n((int*)local_acc_vec, K * 5, 0);
        #pragma omp for collapse(2)
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                int base_index = W * i + j;
                int img_base_index = 3 * base_index;

                cluster_no_t cluster_no = (cluster_no_t)(assignment[base_index]);
                if (cluster_no == 0xFFFF) continue;
                local_num_cluster_members[cluster_no]++;
                local_acc_vec[cluster_no][0] += i;
                local_acc_vec[cluster_no][1] += j;
                local_acc_vec[cluster_no][2] += image[img_base_index];
                local_acc_vec[cluster_no][3] += image[img_base_index + 1];
                local_acc_vec[cluster_no][4] += image[img_base_index + 2];
            }
        }

        #pragma omp critical
        {
            for (int k = 0; k < K; k++) {
                for (int dim = 0; dim < 5; dim++) {
                    cluster_acc_vec[k][dim] += local_acc_vec[k][dim];
                }
                num_cluster_members[k] += local_num_cluster_members[k];
            }
        }
    }


    for (int k = 0; k < K; k++) {
        int num_current_members = num_cluster_members[k];
        Cluster *cluster = &clusters[k];
        cluster->num_members = num_current_members;

        if (num_current_members == 0) continue;

        // Technically speaking, as for L1 norm, you need median instead of mean for correct maximization.
        // But, I intentionally used mean here for the sake of performance.
        cluster->y = round_int(cluster_acc_vec[k][0], num_current_members);
        cluster->x = round_int(cluster_acc_vec[k][1], num_current_members);
        cluster->r = round_int(cluster_acc_vec[k][2], num_current_members);
        cluster->g = round_int(cluster_acc_vec[k][3], num_current_members);
        cluster->b = round_int(cluster_acc_vec[k][4], num_current_members);
    }
}


extern "C" {

    void slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters) {

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
                    fast_abs(image[img_base_index + 3] - image[img_base_index - 3]) +
                    fast_abs(image[img_base_index + 4] - image[img_base_index - 2]) +
                    fast_abs(image[img_base_index + 5] - image[img_base_index - 1]);
                int dy = 
                    fast_abs(image[img_base_index + 3 * W] - image[img_base_index - 3 * W]) +
                    fast_abs(image[img_base_index + 3 * W + 1] - image[img_base_index - 3 * W + 1]) +
                    fast_abs(image[img_base_index + 3 * W + 2] - image[img_base_index - 3 * W + 2]);
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

    static void slic_enforce_connectivity(int H, int W, int K, const Cluster* clusters, uint32_t* assignment) {
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

    void do_slic(int H, int W, int K, uint8_t compactness_shift, uint8_t quantize_level, int max_iter, const uint8_t *__restrict__ image, Cluster *__restrict__ clusters, uint32_t* __restrict__ assignment) {
        Context context;
        init_context(&context);
        context.image = image;
        context.algorithm = "cluster_oriented";
        context.H = H;
        context.W = W;
        context.K = K;
        context.S = (int16_t)sqrt(H * W / K);
        context.compactness_shift = compactness_shift;
        context.quantize_level = quantize_level;
        context.clusters = clusters;
        context.assignment = assignment;

        uint32_t quad_image_memory_width;
        context.quad_image_memory_width = quad_image_memory_width = simd_helper::align_to_next(W * 4);

        uint8_t* aligned_quad_image = simd_helper::alloc_aligned_array<uint8_t>(H * quad_image_memory_width);
        for (int i = 0; i < H ; i++) {
            for (int j = 0; j < W; j++) {
                for (int k = 0; k < 3; k++) {
                    aligned_quad_image[i * quad_image_memory_width + 4 * j + k] = image[i * W * 3 + 3 * j + k];
                }
                aligned_quad_image[i * quad_image_memory_width + 4 * j + 3] = 0;
            }
        }
        context.aligned_quad_image = aligned_quad_image;

        for (int i = 0; i < max_iter; i++) {
            auto t1 = Clock::now();
            slic_assign(&context);
            auto t2 = Clock::now();
            slic_update_clusters(&context);
            auto t3 = Clock::now();
            std::cerr << "assignment " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
            std::cerr << "update "<< std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() << "us \n";
        }

        auto t1 = Clock::now();
        slic_enforce_connectivity(H, W, K, clusters, assignment);
        auto t2 = Clock::now();

        std::cerr << "enforce connectivity "<< std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";


        free_context(&context);
    }
}

#ifdef PROTOTYPE_MAIN_DEMO
#include <string>
#include <chrono>
#include <fstream>
#include <memory>
#include <iostream>
typedef std::chrono::high_resolution_clock Clock;
int main(int argc, char** argv) {
    int K = 100;
    int compactness = 5;
    int max_iter = 2;
    int quantize_level = 7;
    try { 
        if (argc > 1) {
            K = std::stoi(std::string(argv[1]));
        }
        if (argc > 2) {
            compactness = std::stoi(std::string(argv[2]));
        }
        if (argc > 3) {
            max_iter = std::stoi(std::string(argv[3]));
        }
        if (argc > 4) {
            quantize_level = std::stoi(std::string(argv[4]));
        }
    } catch (...) {
        std::cerr << "slic num_components compactness max_iter quantize_level" << std::endl;
        return 2;
    }

    int H = 480;
    int W = 640;
    Cluster clusters[K];
    std::unique_ptr<uint8_t[]> image { new uint8_t[H * W * 3] };
    std::unique_ptr<uint32_t[]> assignment { new uint32_t[H * W] };

    std::ifstream inputf("/tmp/a.txt");
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int r, g, b;
            inputf >> r >> g >> b;
            image.get()[3 * W * i + 3 * j] = r;
            image.get()[3 * W * i + 3 * j + 1] = g;
            image.get()[3 * W * i + 3 * j + 2] = b;
        }
    }

    auto t1 = Clock::now();
    slic_initialize_clusters(H, W, K, image.get(), clusters);
    do_slic(H, W, K, compactness, quantize_level, max_iter, image.get(), clusters, assignment.get());

    auto t2 = Clock::now();
    // 6 times faster than skimage.segmentation.slic
    std::cerr << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";

    {
        std::ofstream outputf("/tmp/b.output.txt");
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                outputf << (short)assignment.get()[W * i + j] << " ";
            }
            outputf << std::endl;
        }
    }
    {
        std::ofstream outputf("/tmp/b.clusters.txt");
        for (int k = 0; k < K; k++) {
            outputf << clusters[k].y << " " << clusters[k].x << " " << clusters[k].num_members << std::endl;
        }
    }
    return 0;
}
#endif
