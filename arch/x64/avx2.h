#include <immintrin.h>
#include "../../context.h"



inline void get_assignment_value_vec(
        const Cluster* cluster, const uint16_t* __restrict__ spatial_dist_patch,
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
    __m128i color_dist_vec__narrow duplicate__narrow;
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
    __m128i color_dist_vec__narrow = packed_sad__narrow;
    #ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    {
        uint16_t shorts[8];
        _mm_storeu_si128((__m128i *)shorts, color_dist_vec__narrow);
        for (int v = 0; v < my_min(8, patch_virtual_width - j); v++) {
            int dr = fast_abs<int>((int)img_quad_row[4 * v + 0] - (int)cluster->r);
            int dg = fast_abs<int>((int)img_quad_row[4 * v + 1] - (int)cluster->g);
            int db= fast_abs<int>((int)img_quad_row[4 * v + 2] - (int)cluster->b);
            int dist = (dr + dg + db) ;
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
                           fast_abs<int>(img_quad_row[4 * v + 2] - cluster->b)))
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


namespace fslic {
    class Context_X64_AVX2 : public ContextSIMD {
        using ContextSIMD::ContextSIMD;

        virtual void assign_cluster(const Cluster *cluster) {
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

            for (int16_t i = fit_to_stride(y_lo) - y_lo; i < patch_height; i += subsample_stride) {
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

    };
};
