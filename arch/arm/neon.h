#include <arm_neon.h>
#include <cassert>
#include "../../context.h"

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
    				const uint16_t* spatial_dist_patch_base_row = spatial_dist_patch + patch_memory_width * i;
    	#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    				assert((long long)spatial_dist_patch_base_row % 32 == 0);
    	#endif
    				// not aligned

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
    				#pragma unroll(4)
    				#pragma GCC unroll(4)
    				for (int j = 0; j < patch_width_multiple8; j += 8) {
    					ASSIGNMENT_VALUE_GETTER_BODY
    					vst1q_u16(min_dist_row, new_min_dist);
    					vst1q_u16(assignment_row, new_assignment);
    				}

    				if (0 < patch_width - patch_width_multiple8) {
    					uint16_t new_min_dists[8], new_assignments[8];
    					int j = patch_width_multiple8;
    					ASSIGNMENT_VALUE_GETTER_BODY
    					vst1q_u16(new_min_dists, new_min_dist);
    					vst1q_u16(new_assignments, new_assignment);
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
