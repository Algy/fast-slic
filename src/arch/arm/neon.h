#include <arm_neon.h>
#include <cassert>
#include "../../context.h"
#include "../../lsc.h"
#include "../../parallel.h"

inline void get_assignment_value_vec(
        const Cluster* cluster,
        const uint8_t* img_quad_row, const uint16_t* spatial_dist_patch_row,
        const uint16_t* min_dist_row, const uint16_t* assignment_row,
        uint16x8_t cluster_number_vec, uint8x16_t cluster_color_vec,
        uint16x8_t& new_min_dist, uint16x8_t& new_assignment
        ) {
    uint16x8_t spatial_dist_vec = vld1q_u16(spatial_dist_patch_row);
    uint8x16_t image_segment = vld1q_u8(img_quad_row);
    uint8x16_t image_segment_2 = vld1q_u8(img_quad_row + 16);

    uint8x16_t abs_segment = vabdq_u8(image_segment, cluster_color_vec);
    uint8x16_t abs_segment_2 = vabdq_u8(image_segment_2, cluster_color_vec);

    uint32x4_t sad_segment = vpaddlq_u16(vpaddlq_u8(abs_segment));
    uint32x4_t sad_segment_2 = vpaddlq_u16(vpaddlq_u8(abs_segment_2));

    uint16x8_t color_dist_vec = vcombine_u16(vmovn_u32(sad_segment), vmovn_u32(sad_segment_2));

    uint16x8_t dist_vec = vaddq_u16(color_dist_vec, spatial_dist_vec);
    uint16x8_t old_assignment = vld1q_u16(assignment_row);
    uint16x8_t old_min_dist = vld1q_u16(min_dist_row);
    new_min_dist = vminq_u16(old_min_dist, dist_vec);

    // 0xFFFF if a[i+15:i] == b[i+15:i], 0x0000 otherwise.
    uint16x8_t mask = vceqq_u16(old_min_dist, new_min_dist);
    // if mask[i+15:i] is not zero, choose a[i+15:i], otherwise choose b[i+15:i]
    new_assignment = vbslq_u16(mask, old_assignment, cluster_number_vec);
}


namespace fslic {
    class Context_ARM_NEON : public ContextSIMD {
        using ContextSIMD::ContextSIMD;

        virtual void assign_clusters(const Cluster **target_clusters, int size) {
            for (int cidx = 0; cidx < size; cidx++)  {
                const Cluster *cluster = target_clusters[cidx];
    			uint16_t cluster_number = cluster->number;
                const uint16_t patch_width = spatial_dist_patch.get_width();
    			const uint16_t patch_width_multiple8 = patch_width & 0xFFF8;

    			const int16_t cluster_y = cluster->y, cluster_x = cluster->x;
    			const int16_t y_lo = cluster_y - S, x_lo = cluster_x - S;

    			uint16x8_t cluster_number_vec = {
    				cluster_number,
    				cluster_number,
    				cluster_number,
    				cluster_number,
    				cluster_number,
    				cluster_number,
    				cluster_number,
    				cluster_number
    			};

    			uint8x16_t cluster_color_vec = {
    				(uint8_t)cluster->r,
    				(uint8_t)cluster->g,
    				(uint8_t)cluster->b,
    				0,
    				(uint8_t)cluster->r,
    				(uint8_t)cluster->g,
    				(uint8_t)cluster->b,
    				0,
    				(uint8_t)cluster->r,
    				(uint8_t)cluster->g,
    				(uint8_t)cluster->b,
    				0,
    				(uint8_t)cluster->r,
    				(uint8_t)cluster->g,
    				(uint8_t)cluster->b,
    				0
    			};
                int16_t patch_height = spatial_dist_patch.get_height();
    			for (int16_t i = fit_to_stride(y_lo) - y_lo; i < patch_height; i += subsample_stride) {
    				const uint16_t* spatial_dist_patch_base_row = spatial_dist_patch.get_row(i);
                    const uint8_t *img_quad_base_row = quad_image.get_row(y_lo + i, 4 * x_lo);
                    uint16_t* assignment_base_row = assignment.get_row(i + y_lo, x_lo);
                    uint16_t* min_dist_base_row = min_dists.get_row(i + y_lo, x_lo);

    	#define ASSIGNMENT_VALUE_GETTER_BODY \
    		uint16x8_t new_min_dist, new_assignment; \
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
    			new_min_dist, \
    			new_assignment \
    		);

    				// (16 + 16)(batch size) / 4(rgba quad) = stride 8
    				for (int j = 0; j < patch_width_multiple8; j += 8) {
    					ASSIGNMENT_VALUE_GETTER_BODY
    					vst1q_u16(min_dist_row, new_min_dist);
    					vst1q_u16(assignment_row, new_assignment);
    				}

                    if (0 < patch_width - patch_width_multiple8) {
                        int j = patch_width_multiple8;
                        int rem = patch_width - patch_width_multiple8;
                        ASSIGNMENT_VALUE_GETTER_BODY

                        uint16x4_t dist_4, assignment_4;
                        if (rem >= 4) {
                            vst1_u16(&min_dist_base_row[j],  vget_low_u16(new_min_dist));
                            vst1_u16(&assignment_base_row[j], vget_low_u16(new_assignment));
                            rem -= 4;
                            j += 4;
                            dist_4 = vget_high_u16(new_min_dist);
                            assignment_4 = vget_high_u16(new_assignment);
                        } else {
                            dist_4 = vget_low_u16(new_min_dist);
                            assignment_4 = vget_low_u16(new_assignment);
                        }

                        switch (rem) {
                            case 3:
                                min_dist_base_row[j] = dist_4[0];
                                assignment_base_row[j] = assignment_4[0];
                                min_dist_base_row[j+1] = dist_4[1];
                                assignment_base_row[j+1] = assignment_4[1];
                                min_dist_base_row[j+2] = dist_4[2];
                                assignment_base_row[j+2] = assignment_4[2];
                                break;
                            case 2:
                                min_dist_base_row[j] = dist_4[0];
                                assignment_base_row[j] = assignment_4[0];
                                min_dist_base_row[j+1] = dist_4[1];
                                assignment_base_row[j+1] = assignment_4[1];
                                break;
                            case 1:
                                min_dist_base_row[j] = dist_4[0];
                                assignment_base_row[j] = assignment_4[0];
                                break;
                        }
                    }
    			}
            }
		}
    };

    inline float32x4_t _float32x4_set1(float v) {
        float32x4_t result = {v, v, v, v};
        return result;
    }

    class ContextLSC_ARM_NEON : public ContextLSC {
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

    			uint16x4_t cluster_number_vec = {cluster_no, cluster_no, cluster_no, cluster_no};


                float32x4_t c_0 = _float32x4_set1(centroid_feats[0][cluster_no]);
                float32x4_t c_1 = _float32x4_set1(centroid_feats[1][cluster_no]);
                float32x4_t c_2 = _float32x4_set1(centroid_feats[2][cluster_no]);
                float32x4_t c_3 = _float32x4_set1(centroid_feats[3][cluster_no]);
                float32x4_t c_4 = _float32x4_set1(centroid_feats[4][cluster_no]);
                float32x4_t c_5 = _float32x4_set1(centroid_feats[5][cluster_no]);
                float32x4_t c_6 = _float32x4_set1(centroid_feats[6][cluster_no]);
                float32x4_t c_7 = _float32x4_set1(centroid_feats[7][cluster_no]);
                float32x4_t c_8 = _float32x4_set1(centroid_feats[8][cluster_no]);
                float32x4_t c_9 = _float32x4_set1(centroid_feats[9][cluster_no]);


                for (int i = y_lo; i < y_hi; i++) {
                    if (!valid_subsample_row(i)) continue;
                    for (int j = x_lo; j < x_hi; j += 4) {
                        float* __restrict min_dist_row = min_dists.get_row(i, j);
                        uint16_t* __restrict assignment_row = assignment.get_row(i, j);
                        int index = W * i + j;

                        float32x4_t f_0 = vld1q_f32(&img_feats[0][index]);
                        float32x4_t d_0 = vsubq_f32(f_0, c_0);

                        float32x4_t f_1 = vld1q_f32(&img_feats[1][index]);
                        float32x4_t d_1 = vsubq_f32(f_1, c_1);

                        float32x4_t f_2 = vld1q_f32(&img_feats[2][index]);
                        float32x4_t d_2 = vsubq_f32(f_2, c_2);

                        float32x4_t f_3 = vld1q_f32(&img_feats[3][index]);
                        float32x4_t d_3 = vsubq_f32(f_3, c_3);

                        float32x4_t f_4 = vld1q_f32(&img_feats[4][index]);
                        float32x4_t d_4 = vsubq_f32(f_4, c_4);

                        float32x4_t f_5 = vld1q_f32(&img_feats[5][index]);
                        float32x4_t d_5 = vsubq_f32(f_5, c_5);

                        float32x4_t f_6 = vld1q_f32(&img_feats[6][index]);
                        float32x4_t d_6 = vsubq_f32(f_6, c_6);

                        float32x4_t f_7 = vld1q_f32(&img_feats[7][index]);
                        float32x4_t d_7 = vsubq_f32(f_7, c_7);

                        float32x4_t f_8 = vld1q_f32(&img_feats[8][index]);
                        float32x4_t d_8 = vsubq_f32(f_8, c_8);

                        float32x4_t f_9 = vld1q_f32(&img_feats[9][index]);
                        float32x4_t d_9 = vsubq_f32(f_9, c_9);

                        float32x4_t dist_vec = vmulq_f32(d_0, d_0);
                        dist_vec = vmlaq_f32(dist_vec, d_1, d_1);
                        dist_vec = vmlaq_f32(dist_vec, d_2, d_2);
                        dist_vec = vmlaq_f32(dist_vec, d_3, d_3);
                        dist_vec = vmlaq_f32(dist_vec, d_4, d_4);
                        dist_vec = vmlaq_f32(dist_vec, d_4, d_4);
                        dist_vec = vmlaq_f32(dist_vec, d_5, d_5);
                        dist_vec = vmlaq_f32(dist_vec, d_6, d_6);
                        dist_vec = vmlaq_f32(dist_vec, d_7, d_7);
                        dist_vec = vmlaq_f32(dist_vec, d_8, d_8);
                        dist_vec = vmlaq_f32(dist_vec, d_9, d_9);

                        float32x4_t old_min_dist = vld1q_f32(min_dist_row);
                        uint16x4_t old_assignment = vld1_u16(assignment_row);
                        float32x4_t new_min_dist = vminq_f32(old_min_dist, dist_vec);

                        // 0xFFFF if a[i+15:i] == b[i+15:i], 0x0000 otherwise.
                        uint16x4_t mask = vmovn_u32(vceqq_f32(old_min_dist, new_min_dist));
                        // if mask[i+15:i] is not zero, choose a[i+15:i], otherwise choose b[i+15:i]
                        uint16x4_t new_assignment = vbsl_u16(mask, old_assignment, cluster_number_vec);

                        int rem = x_hi - j;
                        if (rem >= 4) {
                            vst1_u16(assignment_row, new_assignment);
                            vst1q_f32(min_dist_row, new_min_dist);
                        } else {
                            uint16_t arr_assignment[4];
                            float arr_dist[4];
                            vst1_u16(arr_assignment, new_assignment);
                            vst1q_f32(arr_dist, new_min_dist);
                            for (int delta = 0; delta < rem; delta++) {
                                assignment_row[delta] = arr_assignment[delta];
                                min_dist_row[delta] = arr_dist[delta];
                            }
                        }
                    }
                }
            }
        }

    	virtual void normalize_features(float * __restrict numers[10], float* __restrict weights, int size) {
            #pragma omp parallel for num_threads(fsparallel::nth())
            for (int i = 0; i < size; i += 4) {
                float32x4_t reciprocal_w = vrecpeq_f32(vld1q_f32(&weights[i]));
                vst1q_f32(&numers[0][i], vmulq_f32(vld1q_f32(&numers[0][i]), reciprocal_w));
                vst1q_f32(&numers[1][i], vmulq_f32(vld1q_f32(&numers[1][i]), reciprocal_w));
                vst1q_f32(&numers[2][i], vmulq_f32(vld1q_f32(&numers[2][i]), reciprocal_w));
                vst1q_f32(&numers[3][i], vmulq_f32(vld1q_f32(&numers[3][i]), reciprocal_w));
                vst1q_f32(&numers[4][i], vmulq_f32(vld1q_f32(&numers[4][i]), reciprocal_w));
                vst1q_f32(&numers[5][i], vmulq_f32(vld1q_f32(&numers[5][i]), reciprocal_w));
                vst1q_f32(&numers[6][i], vmulq_f32(vld1q_f32(&numers[6][i]), reciprocal_w));
                vst1q_f32(&numers[7][i], vmulq_f32(vld1q_f32(&numers[7][i]), reciprocal_w));
                vst1q_f32(&numers[8][i], vmulq_f32(vld1q_f32(&numers[8][i]), reciprocal_w));
                vst1q_f32(&numers[9][i], vmulq_f32(vld1q_f32(&numers[9][i]), reciprocal_w));
            }
        }
    };
};
