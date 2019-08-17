#include <immintrin.h>
#include "../../context.h"



inline void get_assignment_value_vec(
        const Cluster* cluster,
        const uint8_t* img_quad_row, const uint16_t* spatial_dist_patch_row,
        const uint16_t* min_dist_row, const uint16_t* assignment_row,
        __m128i cluster_number_vec, __m256i cluster_color_vec,
        __m128i order_swap_mask,
        __m128i& new_min_dist__narrow, __m128i& new_assignment__narrow
        ) {
    // image_segment: 16 elements of uint16_t
    // cluster_color_vec: cluster_R cluster_G cluster_B 0 cluster_R cluster_G cluster_B 0 ....
    // spatial_dist_vec: 8 elements of uint16_t (128-bit narrow vector)
    // color_dist_vec, dist_vec: 8 elements of uint16_t (128-bit narrow-vectors)

    __m128i spatial_dist_vec = _mm_load_si128((__m128i *)spatial_dist_patch_row);

    __m256i image_segment = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)img_quad_row));
    __m256i image_segment_2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(img_quad_row + 16)));

    // [R1, G1, B1, A1, R2, G2, B2, A2, R3, G3, B3, A3, R3, G3, B3, A3]
    __m256i abd_segment = _mm256_abs_epi16(_mm256_subs_epi16(image_segment, cluster_color_vec));
    // [R5, G5, B5, A5, R6, G6, B6, A6, R7, G7, B7, A7, R8, G8, B8, A8]
    __m256i abd_segment_2 = _mm256_abs_epi16(_mm256_subs_epi16(image_segment_2, cluster_color_vec));

    // [
    //     R1 + G1,
    //     B1 + A1,
    //     R2 + G2,
    //     B2 + A2,
    //
    //     R5 + G5,
    //     B5 + A5,
    //     R6 + G6,
    //     B6 + A6,

    //     R3 + G3,
    //     B3 + A3,
    //     R4 + G4,
    //     B4 + A4,

    //     R7 + G7,
    //     B7 + A7,
    //     R8 + G8,
    //     B8 + A8,
    //
    // ]
    __m256i pkd_hpair = _mm256_hadds_epi16(abd_segment, abd_segment_2);

    // [
    //     R1 + G1 + B1 + A1, (16-bit)
    //     R2 + G2 + B2 + A2,
    //     R5 + G5 + B5 + A5,
    //     R6 + G6 + B6 + A6,
    //     R3 + G3 + B3 + A3,
    //     R4 + G4 + B4 + A4,
    //     R7 + G7 + B7 + A7,
    //     R8 + G8 + B8 + A8,
    // ]
    __m128i pkd_qpair = _mm_hadds_epi16(
        _mm256_extracti128_si256(pkd_hpair, 0),
        _mm256_extracti128_si256(pkd_hpair, 1)
    );
    // [0 (32-bit), 1, 2, 3] -> [0, 2, 1, 3]
    __m128i color_dist_vec = _mm_castps_si128(
        _mm_permutevar_ps(_mm_castsi128_ps(pkd_qpair), order_swap_mask)
    );
    __m128i dist_vec = _mm_adds_epu16(color_dist_vec, spatial_dist_vec);

    __m128i old_assignment__narrow = _mm_loadu_si128((__m128i *)assignment_row);
    __m128i old_min_dist__narrow = _mm_loadu_si128((__m128i *)min_dist_row);
    new_min_dist__narrow = _mm_min_epu16(old_min_dist__narrow, dist_vec);
    // 0xFFFF if a[i+15:i] == b[i+15:i], 0x0000 otherwise.
    __m128i mask__narrow = _mm_cmpeq_epi16(old_min_dist__narrow, new_min_dist__narrow);
    // if mask[i+7:i] == 0xFF, choose b[i+7:i], otherwise choose a[i+7:i]
    new_assignment__narrow = _mm_blendv_epi8(cluster_number_vec, old_assignment__narrow, mask__narrow);
}


namespace fslic {
    class Context_X64_AVX2 : public ContextSIMD {
        using ContextSIMD::ContextSIMD;

        virtual void assign_clusters(const Cluster **target_clusters, int size) {
            __m128i order_swap_mask = _mm_set_epi32(3, 1, 2, 0); // [0, 1, 2, 3]

            int16_t patch_height = spatial_dist_patch.get_height();
            for (int cidx = 0; cidx < size; cidx++)  {
                const Cluster *cluster = target_clusters[cidx];
                uint16_t cluster_number = cluster->number;
                const uint16_t patch_width = spatial_dist_patch.get_width();
                const uint16_t patch_width_multiple8 = patch_width & 0xFFF8;

                const int16_t cluster_y = cluster->y, cluster_x = cluster->x;
                const int16_t y_lo = cluster_y - S, x_lo = cluster_x - S;

                // Note: x86-64 is little-endian arch. ABGR order is correct.
                const uint64_t cluster_color_quad = ((uint64_t)cluster->b << 32) + ((uint64_t)cluster->g << 16) + ((uint64_t)cluster->r);
                __m256i cluster_color_vec = _mm256_set1_epi64x(cluster_color_quad);
                // 16 elements uint16_t (among there elements are the first 8 elements used)
                __m128i cluster_number_vec = _mm_set1_epi16(cluster_number);

                for (int16_t i = fit_to_stride(y_lo) - y_lo; i < patch_height; i += subsample_stride) {
                    const uint16_t* spatial_dist_patch_base_row = spatial_dist_patch.get_row(i);
        #ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
                    assert((long long)spatial_dist_patch_base_row % 32 == 0);
        #endif
                    // not aligned
                    const uint8_t *img_quad_base_row = quad_image.get_row(y_lo + i, 4 * x_lo);
                    uint16_t* assignment_base_row = assignment.get_row(i + y_lo, x_lo);
                    uint16_t* min_dist_base_row = min_dists.get_row(i + y_lo, x_lo);

        #define ASSIGNMENT_VALUE_GETTER_BODY \
            __m128i new_assignment__narrow, new_min_dist__narrow; \
            uint16_t* min_dist_row = min_dist_base_row + j; /* unaligned */ \
            uint16_t* assignment_row = assignment_base_row + j;  /* unaligned */ \
            const uint8_t* img_quad_row = img_quad_base_row + 4 * j; /*Image rows are not aligned due to x_lo*/ \
            const uint16_t* spatial_dist_patch_row = (uint16_t *)HINT_ALIGNED_AS(spatial_dist_patch_base_row + j, 16); /* Spatial distance patch is aligned */ \
            get_assignment_value_vec( \
                cluster, \
                img_quad_row, \
                spatial_dist_patch_row, \
                min_dist_row, \
                assignment_row, \
                cluster_number_vec, \
                cluster_color_vec, \
                order_swap_mask, \
                new_min_dist__narrow, \
                new_assignment__narrow \
            ); \

                    // 32(batch size) / 4(rgba quad) = stride 8
                    #pragma unroll(4)
                    #pragma GCC unroll(4)
                    for (int j = 0; j < patch_width_multiple8; j += 8) {
                        ASSIGNMENT_VALUE_GETTER_BODY
                        _mm_storeu_si128((__m128i*)min_dist_row, new_min_dist__narrow);
                        _mm_storeu_si128((__m128i*)assignment_row, new_assignment__narrow);
                    }

                    if (0 < patch_width - patch_width_multiple8) {
                        uint16_t new_min_dists[8], new_assignments[8];
                        int j = patch_width_multiple8;
                        ASSIGNMENT_VALUE_GETTER_BODY
                        _mm_storeu_si128((__m128i*)new_min_dists, new_min_dist__narrow);
                        _mm_storeu_si128((__m128i*)new_assignments, new_assignment__narrow);

                        for (int delta = 0; delta < patch_width - patch_width_multiple8; delta++) {
                            min_dist_row[delta] = new_min_dists[delta];
                            assignment_row[delta] = new_assignments[delta];
                        }
                    }
                }

            }
        }

    };
};
