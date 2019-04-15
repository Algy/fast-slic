#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <cstring>
#include <cassert>
#include <iostream>
#include <iomanip>

#include <immintrin.h>
#include "fast-slic-avx2.h"
#include "simd-helper.hpp"

#define CHARBIT 8

#ifdef _MSC_VER
#define __restrict__ __restrict
#endif

#ifdef USE_AVX2
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

struct Context {
    const uint8_t* __restrict__ image;
    uint8_t* __restrict__ aligned_quad_image; // copied image
    uint16_t quad_image_memory_width;

    const char* algorithm;
    int H;
    int W;
    int K;
    int16_t S;
    uint8_t compactness;
    uint8_t quantize_level;
    Cluster* __restrict__ clusters;
    uint32_t* __restrict__ aligned_assignment;
    int assignment_memory_width; // memory width of aligned_assignment
    uint16_t* __restrict__ spatial_normalize_cache; // (x) -> (uint16_t)(((uint32_t)x << quantize_level) * M / S / 2 * 3) 
    uint16_t* __restrict__ spatial_dist_patch;
};

static void init_context(Context *context) {
    memset(context, 0, sizeof(Context));
}

static void prepare_spatial(Context *context) {
    int16_t S = context->S;
    int8_t quantize_level = context->quantize_level;
    uint8_t compactness = context->compactness;

    uint16_t* spatial_normalize_cache = new uint16_t[2 * S + 2];
    context->spatial_normalize_cache = spatial_normalize_cache;
    for (int x = 0; x < 2 * S + 2; x++) {
        context->spatial_normalize_cache[x] = (uint16_t)(((uint32_t)x * compactness << quantize_level) / S / 2 * 3);
    }

    const uint16_t patch_height = 2 * S + 1, patch_virtual_width = 2 * S + 1;
    const uint16_t patch_memory_width = simd_helper::align_to_next(patch_virtual_width);

    uint16_t* spatial_dist_patch = simd_helper::alloc_aligned_array<uint16_t>(patch_height * patch_memory_width);
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
    context->spatial_dist_patch = spatial_dist_patch;
}

static void free_context(Context *context) {
    // std::cerr << "Freeing Spatial normalize cache" <<std::endl;
    if (context->spatial_normalize_cache)
        delete [] context->spatial_normalize_cache;
    // std::cerr << "Freeing Aligned quad image" <<std::endl;
    if (context->aligned_quad_image) {
        simd_helper::free_aligned_array(context->aligned_quad_image);
    }
    // std::cerr << "Freeing Aligned assignment" <<std::endl;
    if (context->aligned_assignment) {
        simd_helper::free_aligned_array(context->aligned_assignment);
    }
    // std::cerr << "Freeing spatial dist patch" <<std::endl;
    if (context->spatial_dist_patch) {
        simd_helper::free_aligned_array(context->spatial_dist_patch);
    }
}

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


struct ZOrderTuple {
    uint32_t score;
    const Cluster* cluster;

    ZOrderTuple(uint32_t score, const Cluster* cluster) : score(score), cluster(cluster) {};
};

bool operator<(const ZOrderTuple &lhs, const ZOrderTuple &rhs) {
    return lhs.score < rhs.score;
}


#include <string>
#include <chrono>
#include <fstream>
#include <memory>
#include <iomanip>
typedef std::chrono::high_resolution_clock Clock;

static inline
__m256i get_assignment_value_vec(
        const Cluster* cluster, uint8_t quantize_level, const uint16_t* __restrict__ spatial_dist_patch,
        int patch_memory_width,
        int i, int j, int patch_virtual_width,
        const uint8_t* img_quad_row, const uint16_t* spatial_dist_patch_row,
        __m256i cluster_number_vec, __m256i cluster_color_vec64, __m256i cluster_color_vec,
        __m256i color_swap_mask, __m128i sad_duplicate_mask
        ) {
#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    assert((long long)(spatial_dist_patch_row) % 16 == 0);
#endif
    // image_segment: 32 elements of uint8_t 
    // cluster_color_vec64: cluster_R cluster_B cluster_G 0 0 0 0 0 cluster_R cluster_G cluster_B 0 ....
    // cluster_color_vec: cluster_R cluster_B cluster_G 0 cluster_R cluster_G cluster_B 0 ...
    // spatial_dist_vec__narrow: 8 elements of uint16_t (128-bit narrow vector)
    // color_dist_vec__narrow, dist_vec__narrow: 8 elements of uint16_t (128-bit narrow-vectors)
    // assignment_value_vec: 
    //   8 elements of uint32_t
    //   [high 16-bit: distance value] + [low 16-bit: cluster_number]

    __m128i spatial_dist_vec__narrow = _mm_load_si128((__m128i *)spatial_dist_patch_row);
#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    {
        uint16_t spatial_dists[8];
        _mm_storeu_si128((__m128i*)spatial_dists, spatial_dist_vec__narrow);
        for (int delta = 0; delta < 8; delta++) {
            assert(spatial_dists[delta] == spatial_dist_patch[patch_memory_width * i + (j + delta)]);
        }
    }
#endif

    __m256i image_segment = _mm256_loadu_si256((__m256i*)img_quad_row);

#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    {
        uint8_t s[32];
        _mm256_storeu_si256((__m256i *)s, image_segment);
        for (int v = 0; v < 4 * my_min(8, patch_virtual_width - j); v++) {
            if (s[v] != img_quad_row[v]) {
                abort();
            }
        }
    }
#endif

#ifdef FAST_SLIC_AVX2_FASTER
    __m256i sad = _mm256_sad_epu8(image_segment, cluster_color_vec);
    __m128i shrinked__narrow = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(sad, color_swap_mask));
    __m128i duplicate__narrow = _mm_shuffle_epi8(shrinked__narrow, sad_duplicate_mask);
    __m128i color_dist_vec__narrow = _mm_slli_epi32(duplicate__narrow, quantize_level);
    __m128i dist_vec__narrow = _mm_adds_epu16(color_dist_vec__narrow, spatial_dist_vec__narrow);
#else 
    __m128i lo_segment__narrow = _mm256_extracti128_si256(image_segment, 0);
    __m128i hi_segment__narrow = _mm256_extracti128_si256(image_segment, 1);

    __m256i lo_segment = _mm256_cvtepu32_epi64(lo_segment__narrow);
    __m256i hi_segment = _mm256_cvtepu32_epi64(hi_segment__narrow);

    __m256i lo_sad = _mm256_sad_epu8(lo_segment, cluster_color_vec64);
    __m256i hi_sad = _mm256_sad_epu8(hi_segment, cluster_color_vec64);

    __m128i lo_sad__narrow = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(lo_sad, color_swap_mask));
    __m128i hi_sad__narrow = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(hi_sad, color_swap_mask));
    __m128i packed_sad__narrow = _mm_packs_epi32(lo_sad__narrow, hi_sad__narrow);
    __m128i color_dist_vec__narrow = _mm_slli_epi32(packed_sad__narrow, quantize_level);
    #ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    {
        uint16_t shorts[8];
        _mm_storeu_si128((__m128i *)shorts, color_dist_vec__narrow);
        for (int v = 0; v < my_min(8, patch_virtual_width - j); v++) {
            int dr = fast_abs<int>((int)img_quad_row[4 * v + 0] - (int)cluster->r);
            int dg = fast_abs<int>((int)img_quad_row[4 * v + 1] - (int)cluster->g);
            int db= fast_abs<int>((int)img_quad_row[4 * v + 2] - (int)cluster->b);
            int dist = (dr + dg + db) << quantize_level;
            assert((int)shorts[v] == dist);
        }
    }
    #endif
    __m128i dist_vec__narrow = _mm_adds_epu16(color_dist_vec__narrow, spatial_dist_vec__narrow);
    #ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
        {
            uint16_t dists[8];
            _mm_storeu_si128((__m128i*)dists, dist_vec__narrow);
            for (int v = 0; v < my_min(8, patch_virtual_width - j); v++) {
                assert(
                        (int)dists[v] ==
                        ((int)spatial_dist_patch[patch_memory_width * i + (j + v)] +
                         ((fast_abs<int>(img_quad_row[4 * v + 0]  - cluster->r) +
                           fast_abs<int>(img_quad_row[4 * v + 1] - cluster->g) +
                           fast_abs<int>(img_quad_row[4 * v + 2] - cluster->b)) << quantize_level)
                        )
                      );
            }
        }
    #endif
#endif
    // __m256i assignment_value_vec = _mm256_unpacklo_epi16(dist_vec__narrow, cluster_number_vec);
    __m256i assignment_value_vec = _mm256_add_epi32(_mm256_slli_epi32(_mm256_cvtepu16_epi32(dist_vec__narrow), 16), cluster_number_vec);
#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    {
        // asm("int $3");
        uint16_t dists[8];
        uint32_t values[8];
        _mm256_storeu_si256((__m256i *)values, assignment_value_vec);
        _mm_storeu_si128((__m128i *)dists, dist_vec__narrow);
        for (int v = 0; v < 8; v++) {
            assert((values[v] & 0xFFFF) == cluster->number);
            assert((values[v] >> 16) == dists[v]);
        }
    }

#endif
    return assignment_value_vec;
}

static void slic_assign_cluster_oriented(Context *context) {
    auto H = context->H;
    auto W = context->W;
    auto K = context->K;
    auto clusters = context->clusters;
    auto assignment_memory_width = context->assignment_memory_width;
    auto quantize_level = context->quantize_level;
    const int16_t S = context->S;

    const uint8_t* __restrict__ aligned_quad_image = (uint8_t* __restrict__)HINT_ALIGNED(context->aligned_quad_image);
    const uint16_t* __restrict__ spatial_dist_patch = (const uint16_t* __restrict__)HINT_ALIGNED(context->spatial_dist_patch);
    uint32_t* __restrict__ aligned_assignment = (uint32_t* __restrict__)HINT_ALIGNED(context->aligned_assignment);

    auto quad_image_memory_width = context->quad_image_memory_width;

    // might help to initialize array
#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    assert((long long)aligned_quad_image % 32 == 0);
    assert((long long)spatial_dist_patch % 32 == 0);
    assert((long long)aligned_assignment % 32 == 0);
#endif

    const uint16_t patch_height = 2 * S + 1, patch_virtual_width = 2 * S + 1;
    const uint16_t patch_memory_width = simd_helper::align_to_next(patch_virtual_width);


    // Sorting clusters by morton order seems to help for distributing clusters evenly for multiple cores
    // auto t0 = Clock::now();

    std::vector<ZOrderTuple> cluster_sorted_tuples;
    {
        cluster_sorted_tuples.reserve(K);
        for (int k = 0; k < K; k++) {
            const Cluster* cluster = &clusters[k];
            uint32_t score = get_sort_value(cluster->y, cluster->x, S);
            cluster_sorted_tuples.push_back(ZOrderTuple(score, cluster));
        }
        std::sort(cluster_sorted_tuples.begin(), cluster_sorted_tuples.end());
    }

    // auto t1 = Clock::now();
    __m256i color_swap_mask =  _mm256_set_epi32(
        7, 7, 7, 7,
        6, 4, 2, 0

    );
    __m128i sad_duplicate_mask = _mm_set_epi8(
        13, 12, 13, 12,
        9, 8, 9, 8,
        5, 4, 5, 4,
        1, 0, 1, 0
    );

 
    #pragma omp parallel for schedule(static)
    for (int cluster_sorted_idx = 0; cluster_sorted_idx < K; cluster_sorted_idx++) {
        const Cluster *cluster = cluster_sorted_tuples[cluster_sorted_idx].cluster;
        cluster_no_t cluster_number = cluster->number;
        const int16_t cluster_y = my_min<int16_t>(my_max<int16_t>(cluster->y, S), H - S - 1), cluster_x = my_min<int16_t>(my_max<int16_t>(cluster->x, S), W - S - 1);
        const int16_t y_lo = cluster_y - S, x_lo = cluster_x - S;

        // Note: x86-64 is little-endian arch. ABGR order is correct.
        const uint32_t cluster_color_quad = ((uint32_t)cluster->b << 16) + ((uint32_t)cluster->g << 8) + ((uint32_t)cluster->r);
        __m256i cluster_color_vec64 = _mm256_set1_epi64x((uint64_t)cluster_color_quad);
        __m256i cluster_color_vec = _mm256_set1_epi32((uint32_t)cluster_color_quad);
        // 16 elements uint16_t (among there elements are the first 8 elements used)
        __m256i cluster_number_vec = _mm256_set1_epi32((uint32_t)cluster_number);

        for (int16_t i = 0; i < patch_height; i++) {
            const uint16_t* spatial_dist_patch_base_row = spatial_dist_patch + patch_memory_width * i;
#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
            assert((long long)spatial_dist_patch_base_row % 32 == 0);
#endif
            // not aligned because of x_lo
            const uint8_t *img_quad_base_row = aligned_quad_image + quad_image_memory_width * (y_lo + i) + 4 * x_lo;
            uint32_t* assignment_base_row = aligned_assignment + (i + y_lo) * assignment_memory_width + x_lo;

#define ASSIGNMENT_VALUE_GETTER_BODY const uint16_t* spatial_dist_patch_row; const uint8_t* img_quad_row; uint32_t* assignment_row; __m256i assignment_value_vec; { \
    img_quad_row = img_quad_base_row + 4 * j; /*Image rows are not aligned due to x_lo*/ \
    spatial_dist_patch_row = spatial_dist_patch_base_row + j; /* Spatial distance patch is aligned */ \
    spatial_dist_patch_row = (uint16_t *)HINT_ALIGNED_AS(spatial_dist_patch_row, 16); \
    assignment_row = assignment_base_row + j; /* unaligned */ \
    assignment_value_vec = get_assignment_value_vec( \
        cluster, \
        quantize_level, \
        spatial_dist_patch, \
        patch_memory_width, \
        i, j, patch_virtual_width, \
        img_quad_row, \
        spatial_dist_patch_row, \
        cluster_number_vec, \
        cluster_color_vec64, \
        cluster_color_vec, \
        color_swap_mask, \
        sad_duplicate_mask \
    ); \
}
            const uint16_t patch_virtual_width_multiple8 = patch_virtual_width & 0xFFF8;
            // 32(batch size) / 4(rgba quad) = stride 8 
            #pragma unroll(4)
            #pragma GCC unroll(4)
            for (int j = 0; j < patch_virtual_width_multiple8; j += 8) {
                ASSIGNMENT_VALUE_GETTER_BODY
                // min-assignment
                // Race condition is here. But who cares?
                __m256i min_assignment_vec = _mm256_min_epu32(_mm256_loadu_si256((__m256i*)assignment_row), assignment_value_vec);
                _mm256_storeu_si256((__m256i*)assignment_row, min_assignment_vec);
            }

            if (patch_virtual_width_multiple8 < patch_virtual_width) {
                int j = patch_virtual_width_multiple8;
                ASSIGNMENT_VALUE_GETTER_BODY
                ALIGN_SIMD uint32_t calcd_values[8];
                _mm256_store_si256((__m256i *)calcd_values, assignment_value_vec);
                const int max_V = patch_virtual_width - j;
                for (int k = 0; k < max_V; k++) {
                    if (assignment_row[k] > calcd_values[k]) {
                        assignment_row[k] = calcd_values[k];
                    }
                }
            }
        }
    }
    // auto t2 = Clock::now();
    // std::cerr << "Sort: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << "us \n";
    // std::cerr << "Tightloop: " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
}

static void slic_assign(Context *context) {
    if (!strcmp(context->algorithm, "cluster_oriented")) {
        slic_assign_cluster_oriented(context);
    }
}

static void slic_update_clusters(Context *context, bool reset_assignment) {
    auto H = context->H;
    auto W = context->W;
    auto K = context->K;
    auto aligned_quad_image = context->aligned_quad_image;
    auto clusters = context->clusters;
    auto aligned_assignment = context->aligned_assignment;
    auto quad_image_memory_width = context->quad_image_memory_width;
    auto assignment_memory_width = context->assignment_memory_width;

    int *num_cluster_members = new int[K];
    int *cluster_acc_vec = new int[K * 5]; // sum of [y, x, r, g, b] in cluster

    std::fill_n(num_cluster_members, K, 0);
    std::fill_n((int *)cluster_acc_vec, K * 5, 0);

    #pragma omp parallel
    {
        uint32_t *local_acc_vec = new uint32_t[K * 5]; // sum of [y, x, r, g, b] in cluster
        int *local_num_cluster_members = new int[K];
        std::fill_n(local_num_cluster_members, K, 0);
        std::fill_n(local_acc_vec, K * 5, 0);

        #if _OPENMP >= 200805
        #pragma omp for collapse(2)
        #else
        #pragma omp for
        #endif
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                int img_base_index = quad_image_memory_width * i + 4 * j;
                int assignment_index = assignment_memory_width * i + j;

                cluster_no_t cluster_no = (cluster_no_t)(aligned_assignment[assignment_index] & 0x0000FFFF);
                if (reset_assignment) aligned_assignment[assignment_index] = 0xFFFFFFFF;
                if (cluster_no != 0xFFFF && cluster_no < K) {
                    local_num_cluster_members[cluster_no]++;
                    local_acc_vec[5 * cluster_no + 0] += i;
                    local_acc_vec[5 * cluster_no + 1] += j;
                    local_acc_vec[5 * cluster_no + 2] += aligned_quad_image[img_base_index];
                    local_acc_vec[5 * cluster_no + 3] += aligned_quad_image[img_base_index + 1];
                    local_acc_vec[5 * cluster_no + 4] += aligned_quad_image[img_base_index + 2];
                }
            }
        }

        #pragma omp critical
        {
            for (int k = 0; k < K; k++) {
                for (int dim = 0; dim < 5; dim++) {
                    cluster_acc_vec[5 * k + dim] += local_acc_vec[5 * k + dim];
                }
                num_cluster_members[k] += local_num_cluster_members[k];
            }
        }

        delete [] local_num_cluster_members;
        delete [] local_acc_vec;
    }


    for (int k = 0; k < K; k++) {
        int num_current_members = num_cluster_members[k];
        Cluster *cluster = &clusters[k];
        cluster->num_members = num_current_members;

        if (num_current_members == 0) continue;

        // Technically speaking, as for L1 norm, you need median instead of mean for correct maximization.
        // But, I intentionally used mean here for the sake of performance.
        cluster->y = round_int(cluster_acc_vec[5 * k + 0], num_current_members);
        cluster->x = round_int(cluster_acc_vec[5 * k + 1], num_current_members);
        cluster->r = round_int(cluster_acc_vec[5 * k + 2], num_current_members);
        cluster->g = round_int(cluster_acc_vec[5 * k + 3], num_current_members);
        cluster->b = round_int(cluster_acc_vec[5 * k + 4], num_current_members);
    }
    delete [] num_cluster_members;
    delete [] cluster_acc_vec;
}


static void slic_enforce_connectivity(int H, int W, int K, const Cluster* clusters, int assignment_memory_width, uint32_t* aligned_assignment) {
    if (K <= 0) return;

    uint8_t *visited = new uint8_t[H * assignment_memory_width];
    std::fill_n(visited, H * W, 0);

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int base_index = assignment_memory_width * i + j;
            if (aligned_assignment[base_index] != 0xFFFF) continue;

            std::vector<int> visited_indices;
            std::vector<int> stack;
            std::unordered_set<int> adj_cluster_indices;
            stack.push_back(base_index);
            while (!stack.empty()) {
                int index = stack.back();
                stack.pop_back();

                if (aligned_assignment[index] != 0xFFFF) {
                    adj_cluster_indices.insert(aligned_assignment[index]);
                    continue;
                } else if (visited[index]) {
                    continue;
                }
                visited[index] = 1;
                visited_indices.push_back(index);

                int index_j = index % W;
                // up
                if (index > assignment_memory_width) {
                    stack.push_back(index - assignment_memory_width);
                }

                // down
                if (index + assignment_memory_width < H * assignment_memory_width) {
                    stack.push_back(index + assignment_memory_width);
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
                aligned_assignment[*it] = target_cluster_index;
            }

        }
    }
    delete [] visited;
}



extern "C" {
    void fast_slic_initialize_clusters_avx2(int H, int W, int K, const uint8_t* image, Cluster *clusters) {

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
    void fast_slic_iterate_avx2(int H, int W, int K, uint8_t compactness, uint8_t quantize_level, int max_iter, const uint8_t *__restrict__ image, Cluster *__restrict__ clusters, uint32_t* __restrict__ assignment) {
        Context context;
        init_context(&context);
        context.image = image;
        context.algorithm = "cluster_oriented";
        context.H = H;
        context.W = W;
        context.K = K;
        context.S = (int16_t)sqrt(H * W / K);
        context.compactness = compactness;
        context.quantize_level = quantize_level;
        context.clusters = clusters;

        uint32_t quad_image_memory_width;
        context.quad_image_memory_width = quad_image_memory_width = simd_helper::align_to_next(W * 4);

        uint8_t* aligned_quad_image = simd_helper::alloc_aligned_array<uint8_t>(H * quad_image_memory_width);
        for (int i = 0; i < H ; i++) {
            for (int j = 0; j < W; j++) {
                for (int k = 0; k < 3; k++) {
                    aligned_quad_image[i * quad_image_memory_width + 4 * j + k] = image[i * W * 3 + 3 * j + k];
                }
            }
        }
        context.aligned_quad_image = aligned_quad_image;
        uint32_t assignment_memory_width = simd_helper::align_to_next(W);
        uint32_t *aligned_assignment = simd_helper::alloc_aligned_array<uint32_t>(H * assignment_memory_width);
        context.assignment_memory_width = assignment_memory_width;
        context.aligned_assignment = aligned_assignment;

        prepare_spatial(&context);

        // Take advantage of aligned_assignment being aligned
        {
            // auto t1 = Clock::now();
            __m256i constant = _mm256_set1_epi32(0xFFFFFFFF);
            #pragma omp parallel for
            for (int i = 0; i < H; i++) {
                #pragma unroll(4)
                #pragma GCC unroll(4)
                for (int j = 0; j < W; j += 8) {
                    _mm256_store_si256((__m256i *)(aligned_assignment + assignment_memory_width * i + j), constant);
                }
            }
            // auto t2 = Clock::now();
            // std::cerr << "Assignment Initialization " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
        }


        for (int i = 0; i < max_iter; i++) {
            // auto t1 = Clock::now();
            slic_assign(&context);
            // auto t2 = Clock::now();
            slic_update_clusters(&context, i + 1 < max_iter);
            // auto t3 = Clock::now();
            // std::cerr << "assignment " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
            // std::cerr << "update "<< std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() << "us \n";
        }

        // auto t1 = Clock::now();
        slic_enforce_connectivity(H, W, K, clusters, assignment_memory_width, aligned_assignment);
        // auto t2 = Clock::now();

        // std::cerr << "enforce connectivity "<< std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";

        {
            // auto t1 = Clock::now();
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    assignment[W * i + j] = context.aligned_assignment[context.assignment_memory_width * i + j] & 0x0000FFFF;
                }
            }
            // auto t2 = Clock::now();
            // std::cerr << "Write back assignment"<< std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
        }

        // std::cerr << "Freeing context" <<std::endl;
        free_context(&context);
        // std::cerr << "Freeing context. done." <<std::endl;
    }
    int fast_slic_supports_avx2() { return 1; }
}

#else // else of #ifdef USE_AVX2

extern "C" {
    void fast_slic_initialize_clusters_avx2(int H, int W, int K, const uint8_t* image, Cluster *clusters) {}
    void fast_slic_iterate_avx2(int H, int W, int K, uint8_t compactness, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint32_t* assignment) {}
int fast_slic_supports_avx2() { return 0; }
}

#endif // of #ifdef USE_AVX2


#ifdef PROTOTYPE_MAIN_DEMO

#ifndef USE_AVX2
#error "Compile it with flag USE_AVX"
#endif 

#include <cstdlib>
#include <ctime>
#include <string>
#include <chrono>
#include <fstream>
#include <memory>
typedef std::chrono::high_resolution_clock Clock;
int main(int argc, char** argv) {
    int K = 100;
    int compactness = 5;
    int max_iter = 2;
    int quantize_level = 6;
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
    srand(time(nullptr));

    Cluster clusters[K];
    std::unique_ptr<uint8_t[]> image { new uint8_t[H * W * 3] };
    std::unique_ptr<uint32_t[]> assignment { new uint32_t[H * W] };

    /*
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
    */
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int r, g, b;
            r = (int)(rand() * 255);
            g = (int)(rand() * 255);
            b = (int)(rand() * 255);
            image.get()[3 * W * i + 3 * j] = r;
            image.get()[3 * W * i + 3 * j + 1] = g;
            image.get()[3 * W * i + 3 * j + 2] = b;
        }
    }

    auto t1 = Clock::now();
    fast_slic_initialize_clusters_avx2(H, W, K, image.get(), clusters);
    fast_slic_iterate_avx2(H, W, K, compactness, quantize_level, max_iter, image.get(), clusters, assignment.get());

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
