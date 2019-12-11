#include <immintrin.h>
#include "../../context.h"
#include "../../lsc.h"
#include "../../parallel.h"


inline __m256 _mm256_set_ps1(float v) {
    return _mm256_set_ps(v, v, v, v, v, v, v, v);
}

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
                    for (int j = 0; j < patch_width_multiple8; j += 8) {
                        ASSIGNMENT_VALUE_GETTER_BODY
                        _mm_storeu_si128((__m128i*)min_dist_row, new_min_dist__narrow);
                        _mm_storeu_si128((__m128i*)assignment_row, new_assignment__narrow);
                    }

                    if (0 < patch_width - patch_width_multiple8) {
                        int j = patch_width_multiple8;
                        int rem = patch_width - patch_width_multiple8;
                        ASSIGNMENT_VALUE_GETTER_BODY

                        uint64_t dist_4, assignment_4;
                        if (rem >= 4) {
                            *(uint64_t *)&min_dist_base_row[j] = _mm_extract_epi64(new_min_dist__narrow, 0);
                            *(uint64_t *)&assignment_base_row[j] = _mm_extract_epi64(new_assignment__narrow, 0);
                            rem -= 4;
                            j += 4;
                            dist_4 = _mm_extract_epi64(new_min_dist__narrow, 1);
                            assignment_4 = _mm_extract_epi64(new_assignment__narrow, 1);
                        } else {
                            dist_4 = _mm_extract_epi64(new_min_dist__narrow, 0);
                            assignment_4 = _mm_extract_epi64(new_assignment__narrow, 0);
                        }

                        switch (rem) {
                        case 3:
                            *(uint32_t *)&min_dist_base_row[j] = (uint32_t)dist_4;
                            *(uint32_t *)&assignment_base_row[j] = (uint32_t)assignment_4;
                            *(uint16_t *)&min_dist_base_row[j + 2] = (uint16_t)(dist_4 >> 32);
                            *(uint16_t *)&assignment_base_row[j + 2] = (uint16_t)(assignment_4 >> 32);
                            break;
                        case 2:
                            *(uint32_t *)&min_dist_base_row[j] = (uint32_t)dist_4;
                            *(uint32_t *)&assignment_base_row[j] = (uint32_t)assignment_4;
                            break;
                        case 1:
                            *(uint16_t *)&min_dist_base_row[j] = (uint16_t)dist_4;
                            *(uint16_t *)&assignment_base_row[j] = (uint16_t)assignment_4;
                            break;
                        }
                    }
                }

            }
        }

    };

    class ContextLSC_X64_AVX2 : public ContextLSC {
    public:
        using ContextLSC::ContextLSC;
    protected:
        virtual void assign_clusters(const Cluster **target_clusters, int size) {
            const float* __restrict img_feats[10];
            const float* __restrict centroid_feats[10];

            for (int i = 0; i < 10; i++) {
                img_feats[i] = &image_features[i][0];
                centroid_feats[i] = &centroid_features[i][0];
            }

            for (int cidx = 0; cidx < size; cidx++) {
                const Cluster* cluster = target_clusters[cidx];
                int cluster_y = cluster->y, cluster_x = cluster->x;
                uint16_t cluster_no = cluster->number;

                int y_lo = my_max<int>(cluster_y - S, 0), y_hi = my_min<int>(cluster_y + S + 1, H);
                int x_lo = my_max<int>(cluster_x - S, 0), x_hi = my_min<int>(cluster_x + S + 1, W);

                __m128i cluster_number_vec = _mm_set1_epi16(cluster_no);

                __m256 c_0 = _mm256_set_ps1(centroid_feats[0][cluster_no]);
                __m256 c_1 = _mm256_set_ps1(centroid_feats[1][cluster_no]);
                __m256 c_2 = _mm256_set_ps1(centroid_feats[2][cluster_no]);
                __m256 c_3 = _mm256_set_ps1(centroid_feats[3][cluster_no]);
                __m256 c_4 = _mm256_set_ps1(centroid_feats[4][cluster_no]);
                __m256 c_5 = _mm256_set_ps1(centroid_feats[5][cluster_no]);
                __m256 c_6 = _mm256_set_ps1(centroid_feats[6][cluster_no]);
                __m256 c_7 = _mm256_set_ps1(centroid_feats[7][cluster_no]);
                __m256 c_8 = _mm256_set_ps1(centroid_feats[8][cluster_no]);
                __m256 c_9 = _mm256_set_ps1(centroid_feats[9][cluster_no]);


                for (int i = y_lo; i < y_hi; i++) {
                    if (!valid_subsample_row(i)) continue;
                    for (int j = x_lo; j < x_hi; j += 8) {
                        float* __restrict min_dist_row = min_dists.get_row(i, j);
                        uint16_t* __restrict assignment_row = assignment.get_row(i, j);
                        int index = W * i + j;
                        __m256 f_0 = _mm256_loadu_ps(&img_feats[0][index]);
                        __m256 d_0 = _mm256_sub_ps(f_0, c_0);

                        __m256 f_1 = _mm256_loadu_ps(&img_feats[1][index]);
                        __m256 d_1 = _mm256_sub_ps(f_1, c_1);

                        __m256 f_2 = _mm256_loadu_ps(&img_feats[2][index]);
                        __m256 d_2 = _mm256_sub_ps(f_2, c_2);

                        __m256 f_3 = _mm256_loadu_ps(&img_feats[3][index]);
                        __m256 d_3 = _mm256_sub_ps(f_3, c_3);

                        __m256 f_4 = _mm256_loadu_ps(&img_feats[4][index]);
                        __m256 d_4 = _mm256_sub_ps(f_4, c_4);

                        __m256 f_5 = _mm256_loadu_ps(&img_feats[5][index]);
                        __m256 d_5 = _mm256_sub_ps(f_5, c_5);

                        __m256 f_6 = _mm256_loadu_ps(&img_feats[6][index]);
                        __m256 d_6 = _mm256_sub_ps(f_6, c_6);

                        __m256 f_7 = _mm256_loadu_ps(&img_feats[7][index]);
                        __m256 d_7 = _mm256_sub_ps(f_7, c_7);

                        __m256 f_8 = _mm256_loadu_ps(&img_feats[8][index]);
                        __m256 d_8 = _mm256_sub_ps(f_8, c_8);

                        __m256 f_9 = _mm256_loadu_ps(&img_feats[9][index]);
                        __m256 d_9 = _mm256_sub_ps(f_9, c_9);

                        __m256 dist_vec = _mm256_mul_ps(d_0, d_0);
                        dist_vec = _mm256_fmadd_ps(d_1, d_1, dist_vec);
                        dist_vec = _mm256_fmadd_ps(d_2, d_2, dist_vec);
                        dist_vec = _mm256_fmadd_ps(d_3, d_3, dist_vec);
                        dist_vec = _mm256_fmadd_ps(d_4, d_4, dist_vec);
                        dist_vec = _mm256_fmadd_ps(d_5, d_5, dist_vec);
                        dist_vec = _mm256_fmadd_ps(d_6, d_6, dist_vec);
                        dist_vec = _mm256_fmadd_ps(d_7, d_7, dist_vec);
                        dist_vec = _mm256_fmadd_ps(d_8, d_8, dist_vec);
                        dist_vec = _mm256_fmadd_ps(d_9, d_9, dist_vec);


                        __m128i old_assignment = _mm_loadu_si128((__m128i *)assignment_row);
                        __m256 old_min_dist = _mm256_loadu_ps(min_dist_row);
                        __m256 new_min_dist = _mm256_min_ps(dist_vec, old_min_dist);
                        // 0xFFFFFFFF if a[i+32:i] == b[i+32:i], 0x0000 otherwise.
                        __m256i mask = _mm256_castps_si256(_mm256_cmp_ps(dist_vec, old_min_dist, _CMP_LT_OS));
                        mask = _mm256_srli_epi32(mask, 16);
                        __m128i mask__narrow = _mm256_extracti128_si256(
                            _mm256_permute4x64_epi64(_mm256_packus_epi32(mask, mask), 0xD8),
                            0
                        );

                        // if mask[i+7:i] == 0xFF, choose b[i+7:i], otherwise choose a[i+7:i]
                        __m128i new_assignment = _mm_blendv_epi8(old_assignment, cluster_number_vec, mask__narrow);

                        int rem = x_hi - j;
                        if (rem >= 8) {
                            _mm_storeu_si128((__m128i*)assignment_row, new_assignment);
                            _mm256_storeu_ps(min_dist_row, new_min_dist);
                        } else {
                            uint16_t arr_assignment[8];
                            float arr_dist[8];
                            _mm_storeu_si128((__m128i*)arr_assignment, new_assignment);
                            _mm256_storeu_ps(arr_dist, new_min_dist);

                            for (int delta = 0; delta < rem; delta++) {
                                assignment_row[delta] = arr_assignment[delta];
                                min_dist_row[delta] = arr_dist[delta];
                            }
                        }
                    }
                }
            }
        }

    	void normalize_features(float * __restrict img_feats[10], float* __restrict weights, int size) {
            #pragma omp parallel for num_threads(fsparallel::nth())
            for (int i = 0; i < size; i += 8) {
                __m256 reciprocal_w = _mm256_rcp_ps(_mm256_loadu_ps(&weights[i]));
                _mm256_storeu_ps(&img_feats[0][i], _mm256_mul_ps(_mm256_loadu_ps(&img_feats[0][i]), reciprocal_w));
                _mm256_storeu_ps(&img_feats[1][i], _mm256_mul_ps(_mm256_loadu_ps(&img_feats[1][i]), reciprocal_w));
                _mm256_storeu_ps(&img_feats[2][i], _mm256_mul_ps(_mm256_loadu_ps(&img_feats[2][i]), reciprocal_w));
                _mm256_storeu_ps(&img_feats[3][i], _mm256_mul_ps(_mm256_loadu_ps(&img_feats[3][i]), reciprocal_w));
                _mm256_storeu_ps(&img_feats[4][i], _mm256_mul_ps(_mm256_loadu_ps(&img_feats[4][i]), reciprocal_w));
                _mm256_storeu_ps(&img_feats[5][i], _mm256_mul_ps(_mm256_loadu_ps(&img_feats[5][i]), reciprocal_w));
                _mm256_storeu_ps(&img_feats[6][i], _mm256_mul_ps(_mm256_loadu_ps(&img_feats[6][i]), reciprocal_w));
                _mm256_storeu_ps(&img_feats[7][i], _mm256_mul_ps(_mm256_loadu_ps(&img_feats[7][i]), reciprocal_w));
                _mm256_storeu_ps(&img_feats[8][i], _mm256_mul_ps(_mm256_loadu_ps(&img_feats[8][i]), reciprocal_w));
                _mm256_storeu_ps(&img_feats[9][i], _mm256_mul_ps(_mm256_loadu_ps(&img_feats[9][i]), reciprocal_w));
            }
        }
    };
};
