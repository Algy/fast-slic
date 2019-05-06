#include <cassert>

#include "fast-slic-avx2.h"
#include "fast-slic-common-impl.hpp"

#ifdef USE_AVX2
#include <immintrin.h>

class Context : public BaseContext {
public:
    uint8_t* __restrict__ aligned_quad_image_base = nullptr;
    uint8_t* __restrict__ aligned_quad_image = nullptr; // copied image
    uint16_t quad_image_memory_width;
    uint16_t* __restrict__ aligned_assignment_base = nullptr;
    uint16_t* __restrict__ aligned_assignment = nullptr;
    uint16_t* __restrict__ aligned_min_dists_base = nullptr;
    uint16_t* __restrict__ aligned_min_dists = nullptr;
    int assignment_memory_width; // memory width of aligned_assignment
    int min_dist_memory_width; // memory width of aligned_min_dists;
public:
    virtual ~Context() {
        if (aligned_quad_image_base) {
            simd_helper::free_aligned_array(aligned_quad_image_base);
        }
        if (aligned_assignment_base) {
            simd_helper::free_aligned_array(aligned_assignment_base);
        }
        if (aligned_min_dists_base) {
            simd_helper::free_aligned_array(aligned_min_dists_base);
        }
    }
};

inline void get_assignment_value_vec(
        const Cluster* cluster, uint8_t quantize_level, const uint16_t* __restrict__ spatial_dist_patch,
        int patch_memory_width,
        int i, int j, int patch_virtual_width,
        const uint8_t* img_quad_row, const uint16_t* spatial_dist_patch_row,
        const uint16_t* min_dist_row, const uint16_t* assignment_row,
        __m128i cluster_number_vec__narrow, __m256i cluster_color_vec64, __m256i cluster_color_vec,
        __m256i color_swap_mask, __m128i sad_duplicate_mask,
        __m128i& new_min_dist__narrow, __m128i& new_assignment__narrow
        ) {
#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    assert((long long)(spatial_dist_patch_row) % 16 == 0);
#endif
    // image_segment: 32 elements of uint8_t 
    // cluster_color_vec64: cluster_R cluster_B cluster_G 0 0 0 0 0 cluster_R cluster_G cluster_B 0 ....
    // cluster_color_vec: cluster_R cluster_B cluster_G 0 cluster_R cluster_G cluster_B 0 ...
    // spatial_dist_vec__narrow: 8 elements of uint16_t (128-bit narrow vector)
    // color_dist_vec__narrow, dist_vec__narrow: 8 elements of uint16_t (128-bit narrow-vectors)

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

    __m128i old_assignment__narrow = _mm_loadu_si128((__m128i *)assignment_row);
    __m128i old_min_dist__narrow = _mm_loadu_si128((__m128i *)min_dist_row);
    new_min_dist__narrow = _mm_min_epu16(old_min_dist__narrow, dist_vec__narrow);
    // 0xFFFF if a[i+15:i] == b[i+15:i], 0x0000 otherwise.
    __m128i mask__narrow = _mm_cmpeq_epi16(old_min_dist__narrow, new_min_dist__narrow);
    // if mask[i+7:i] == 0xFF, choose b[i+7:i], otherwise choose a[i+7:i]
    new_assignment__narrow = _mm_blendv_epi8(cluster_number_vec__narrow, old_assignment__narrow, mask__narrow);
    #ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    {
        uint16_t cluster_arr[8];
        uint16_t old_dists[8], calcd_dists[8], new_dists[8];
        uint16_t old_assignment[8], new_assignment[8];

        _mm_storeu_si128((__m128i *)cluster_arr, cluster_number_vec__narrow);
        _mm_storeu_si128((__m128i *) old_dists, old_min_dist__narrow);
        _mm_storeu_si128((__m128i *) calcd_dists, dist_vec__narrow);
        _mm_storeu_si128((__m128i *) new_dists, new_min_dist__narrow);
        _mm_storeu_si128((__m128i *) old_assignment, old_assignment__narrow);
        _mm_storeu_si128((__m128i *) new_assignment, new_assignment__narrow);

        for (int delta = 0; delta < 8; delta++) {
            assert(cluster_arr[delta] == cluster->number);
        }
        for (int delta = 0; delta < 8; delta++) {
            if (old_dists[delta] > calcd_dists[delta]) {
                assert(new_assignment[delta] == cluster->number);
                assert(new_dists[delta] == calcd_dists[delta]);
            } else {
                assert(new_assignment[delta] == old_assignment[delta]);
                assert(new_dists[delta] == old_dists[delta]);
            }
        }
    }
    #endif
}

static void slic_assign_cluster_oriented(Context *context) {
    auto H = context->H;
    auto W = context->W;
    auto K = context->K;
    auto clusters = context->clusters;
    auto assignment_memory_width = context->assignment_memory_width;
    auto min_dist_memory_width = context->min_dist_memory_width;
    auto quantize_level = context->quantize_level;
    const int16_t S = context->S;

    const uint8_t* __restrict__ aligned_quad_image = context->aligned_quad_image;
    const uint16_t* __restrict__ spatial_dist_patch = (const uint16_t* __restrict__)HINT_ALIGNED(context->spatial_dist_patch);
    uint16_t* __restrict__ aligned_assignment = context->aligned_assignment;
    uint16_t* __restrict__ aligned_min_dists = context->aligned_min_dists;

    auto quad_image_memory_width = context->quad_image_memory_width;

    // might help to initialize array
#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    assert((long long)spatial_dist_patch % 32 == 0);
#endif

    const uint16_t patch_height = 2 * S + 1, patch_virtual_width = 2 * S + 1;
    const uint16_t patch_memory_width = simd_helper::align_to_next(patch_virtual_width);

    __m256i constant = _mm256_set1_epi16(0xFFFF);
    #pragma omp parallel for
    for (int i = 0; i < H; i++) {
        #pragma unroll(4)
        #pragma GCC unroll(4)
        for (int j = 0; j < W; j += 16) {
            _mm256_storeu_si256((__m256i *)&aligned_min_dists[min_dist_memory_width * i + j], constant);
        }
    }

    // Sorting clusters by morton order seems to help for distributing clusters evenly for multiple cores
#   ifdef FAST_SLIC_TIMER
    auto t0 = Clock::now();
#   endif

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

#   ifdef FAST_SLIC_TIMER
    auto t1 = Clock::now();
#   endif
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
        uint16_t cluster_number = cluster->number;
        const uint16_t patch_virtual_width_multiple8 = patch_virtual_width & 0xFFF8;

        const int16_t cluster_y = cluster->y, cluster_x = cluster->x;
        const int16_t y_lo = cluster_y - S, x_lo = cluster_x - S;

        // Note: x86-64 is little-endian arch. ABGR order is correct.
        const uint32_t cluster_color_quad = ((uint32_t)cluster->b << 16) + ((uint32_t)cluster->g << 8) + ((uint32_t)cluster->r);
        __m256i cluster_color_vec64 = _mm256_set1_epi64x((uint64_t)cluster_color_quad);
        __m256i cluster_color_vec = _mm256_set1_epi32((uint32_t)cluster_color_quad);
        // 16 elements uint16_t (among there elements are the first 8 elements used)
        __m128i cluster_number_vec__narrow = _mm_set1_epi16(cluster_number);

        for (int16_t i = 0; i < patch_height; i++) {
            const uint16_t* spatial_dist_patch_base_row = spatial_dist_patch + patch_memory_width * i;
#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
            assert((long long)spatial_dist_patch_base_row % 32 == 0);
#endif
            // not aligned
            const uint8_t *img_quad_base_row = aligned_quad_image + quad_image_memory_width * (y_lo + i) + 4 * x_lo;
            uint16_t* assignment_base_row = aligned_assignment + (i + y_lo) * assignment_memory_width + x_lo;
            uint16_t* min_dist_base_row = aligned_min_dists + (i + y_lo) * min_dist_memory_width + x_lo;

#define ASSIGNMENT_VALUE_GETTER_BODY \
    __m128i new_assignment__narrow, new_min_dist__narrow; \
    uint16_t* min_dist_row = min_dist_base_row + j; /* unaligned */ \
    uint16_t* assignment_row = assignment_base_row + j;  /* unaligned */ \
    const uint8_t* img_quad_row = img_quad_base_row + 4 * j; /*Image rows are not aligned due to x_lo*/ \
    const uint16_t* spatial_dist_patch_row = (uint16_t *)HINT_ALIGNED_AS(spatial_dist_patch_base_row + j, 16); /* Spatial distance patch is aligned */ \
    get_assignment_value_vec( \
        cluster, \
        quantize_level, \
        spatial_dist_patch, \
        patch_memory_width, \
        i, j, patch_virtual_width, \
        img_quad_row, \
        spatial_dist_patch_row, \
        min_dist_row, \
        assignment_row, \
        cluster_number_vec__narrow, \
        cluster_color_vec64, \
        cluster_color_vec, \
        color_swap_mask, \
        sad_duplicate_mask, \
        new_min_dist__narrow, \
        new_assignment__narrow \
    ); \

            // 32(batch size) / 4(rgba quad) = stride 8 
            #pragma unroll(4)
            #pragma GCC unroll(4)
            for (int j = 0; j < patch_virtual_width_multiple8; j += 8) {
                ASSIGNMENT_VALUE_GETTER_BODY
                _mm_storeu_si128((__m128i*)min_dist_row, new_min_dist__narrow);
                _mm_storeu_si128((__m128i*)assignment_row, new_assignment__narrow);
            }

            if (0 < patch_virtual_width - patch_virtual_width_multiple8) {
                uint16_t new_min_dists[8], new_assignments[8];
                int j = patch_virtual_width_multiple8;
                ASSIGNMENT_VALUE_GETTER_BODY
                _mm_storeu_si128((__m128i*)new_min_dists, new_min_dist__narrow);
                _mm_storeu_si128((__m128i*)new_assignments, new_assignment__narrow);

                for (int delta = 0; delta < patch_virtual_width - patch_virtual_width_multiple8; delta++) {
                    min_dist_row[delta] = new_min_dists[delta];
                    assignment_row[delta] = new_assignments[delta];
                }
            }
        }
    }
#   ifdef FAST_SLIC_TIMER
    auto t2 = Clock::now();
    std::cerr << "Sort: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << "us \n";
    std::cerr << "Tightloop: " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
#   endif
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

                uint16_t cluster_no = aligned_assignment[assignment_index];
                if (cluster_no == 0xFFFF) continue;
                local_num_cluster_members[cluster_no]++;
                local_acc_vec[5 * cluster_no + 0] += i;
                local_acc_vec[5 * cluster_no + 1] += j;
                local_acc_vec[5 * cluster_no + 2] += aligned_quad_image[img_base_index];
                local_acc_vec[5 * cluster_no + 3] += aligned_quad_image[img_base_index + 1];
                local_acc_vec[5 * cluster_no + 4] += aligned_quad_image[img_base_index + 2];
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

extern "C" {
    void fast_slic_initialize_clusters_avx2(int H, int W, int K, const uint8_t* image, Cluster *clusters) {
#       ifdef FAST_SLIC_TIMER
        auto t1 = Clock::now();
#       endif
        do_fast_slic_initialize_clusters(H, W, K, image, clusters);
#       ifdef FAST_SLIC_TIMER
        auto t2 = Clock::now();
        std::cerr << "Cluster Initialization: " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
#       endif
    }

    void fast_slic_iterate_avx2(int H, int W, int K, float compactness, float min_size_factor, uint8_t quantize_level, int max_iter, const uint8_t *__restrict__ image, Cluster *__restrict__ clusters, uint16_t* __restrict__ assignment) {
        int S = sqrt(H * W / K);

        Context context;
        context.image = image;
        context.algorithm = "cluster_oriented";
        context.H = H;
        context.W = W;
        context.K = K;
        context.S = (int16_t)S;
        context.assignment = assignment;
        context.compactness = compactness;
        context.min_size_factor = min_size_factor;
        context.quantize_level = quantize_level;
        context.clusters = clusters;

        // Pad image and assignment
        uint32_t quad_image_memory_width;
        context.quad_image_memory_width = quad_image_memory_width = simd_helper::align_to_next((W + 2 * S) * 4);
        uint8_t* aligned_quad_image_base = simd_helper::alloc_aligned_array<uint8_t>((H + 2 * S) * quad_image_memory_width);


        context.aligned_quad_image_base = aligned_quad_image_base;
        context.aligned_quad_image = &aligned_quad_image_base[quad_image_memory_width * S + S * 4];

        context.assignment_memory_width = simd_helper::align_to_next(W + 2 * S);
        context.aligned_assignment_base = simd_helper::alloc_aligned_array<uint16_t>((H + 2 * S) * context.assignment_memory_width);
        context.aligned_assignment = &context.aligned_assignment_base[S * context.assignment_memory_width + S];

        context.min_dist_memory_width = context.assignment_memory_width;
        context.aligned_min_dists_base = simd_helper::alloc_aligned_array<uint16_t>((H + 2 * S) * context.min_dist_memory_width);
        context.aligned_min_dists = &context.aligned_min_dists_base[S * context.min_dist_memory_width + S];

        context.prepare_spatial();
        {
#           ifdef FAST_SLIC_TIMER
            auto t1 = Clock::now();
#           endif

            #pragma omp parallel for
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    for (int k = 0; k < 3; k++) {
                        aligned_quad_image_base[(i + S) * quad_image_memory_width + 4 * (j + S) + k] = image[i * W * 3 + 3 * j + k];
                    }
                }
            }

            __m256i constant = _mm256_set1_epi16(0xFFFF);
            #pragma omp parallel for
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j += 16) {
                    _mm256_storeu_si256((__m256i *)&context.aligned_assignment[context.assignment_memory_width * i + j], constant);
                }
            }
#           ifdef FAST_SLIC_TIMER
            auto t2 = Clock::now();
            std::cerr << "Assignment Initialization " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
#           endif
        }


        for (int i = 0; i < max_iter; i++) {
#           ifdef FAST_SLIC_TIMER
            auto t1 = Clock::now();
#           endif
            slic_assign(&context);
#           ifdef FAST_SLIC_TIMER
            auto t2 = Clock::now();
#           endif
            slic_update_clusters(&context);
#           ifdef FAST_SLIC_TIMER
            auto t3 = Clock::now();
            std::cerr << "assignment " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
            std::cerr << "update "<< std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() << "us \n";
#           endif
        }



        {
#           ifdef FAST_SLIC_TIMER
            auto t1 = Clock::now();
#           endif

            #pragma omp parallel for
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    assignment[W * i + j] = context.aligned_assignment[context.assignment_memory_width * i + j];
                }
            }
#           ifdef FAST_SLIC_TIMER
            auto t2 = Clock::now();
            std::cerr << "Write back assignment"<< std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
#           endif
        }
#       ifdef FAST_SLIC_TIMER
        auto t1 = Clock::now();
#       endif
        slic_enforce_connectivity(&context);
#       ifdef FAST_SLIC_TIMER
        auto t2 = Clock::now();
        std::cerr << "enforce connectivity "<< std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
#       endif
    }
    int fast_slic_supports_avx2() { return 1; }
}

#else // else of #ifdef USE_AVX2

extern "C" {
    void fast_slic_initialize_clusters_avx2(int H, int W, int K, const uint8_t* image, Cluster *clusters) {}
    void fast_slic_iterate_avx2(int H, int W, int K, float compactness, float min_size_factor, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint16_t* assignment) {}
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
    std::unique_ptr<uint16_t[]> assignment { new uint16_t[H * W] };

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
    fast_slic_iterate_avx2(H, W, K, compactness, 0.1, quantize_level, max_iter, image.get(), clusters, assignment.get());

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
