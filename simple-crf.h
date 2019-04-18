#ifndef _SIMPLE_CRF_H
#define _SIMPLE_CRF_H

#include <stdint.h>
#include <stddef.h>
#include "fast-slic-common.h"

struct SimpleCRF;
struct SimpleCRFFrame;

typedef struct SimpleCRFParams {
    float spatial_w;
    float temporal_w;
    float spatial_srgb;
    float temporal_srgb;
    float spatial_sxy;
    float spatial_smooth_w;
    float spatial_smooth_sxy;
} SimpleCRFParams;


typedef struct SimpleCRF* simple_crf_t;
typedef struct SimpleCRFFrame* simple_crf_frame_t;
typedef int simple_crf_time_t;

#ifdef __cplusplus
extern "C" {
#endif
simple_crf_t simple_crf_new(size_t num_classes, size_t num_nodes);
void simple_crf_initialize(simple_crf_t crf);
void simple_crf_free(simple_crf_t crf);

SimpleCRFParams simple_crf_get_params(simple_crf_t crf);
void simple_crf_set_params(simple_crf_t crf, SimpleCRFParams params);
void simple_crf_set_compat(simple_crf_t crf, int cls, float compat_value);
float simple_crf_get_compat(simple_crf_t crf, int cls);

simple_crf_time_t simple_crf_first_time(simple_crf_t crf);
simple_crf_time_t simple_crf_last_time(simple_crf_t crf);
size_t simple_crf_num_time_frames(simple_crf_t crf);
simple_crf_time_t simple_crf_pop_time_frame(simple_crf_t crf);
simple_crf_frame_t simple_crf_push_time_frame(simple_crf_t crf);
simple_crf_frame_t simple_crf_time_frame(simple_crf_t crf, simple_crf_time_t time);
simple_crf_time_t simple_crf_frame_get_time(simple_crf_frame_t frame);

/*
 * Pairwise information setter
 */

// const Cluster[] : shape [num_nodes]
void simple_crf_frame_set_clusters(simple_crf_frame_t frame, const Cluster* clusters);
void simple_crf_frame_set_connectivity(simple_crf_frame_t frame, const Connectivity* conn);

/*
 * Unary Getter/Setter
 */ 

// classes: int[] of shape [num_nodes]
void simple_crf_frame_set_mask(simple_crf_frame_t frame, const int* classes, float confidence);
// probas: float[] of shape [num_classes, num_nodes]
void simple_crf_frame_set_proba(simple_crf_frame_t frame, const float* probas);
void simple_crf_frame_set_unbiased(simple_crf_frame_t frame);

// unary_energies: float[] of shape [num_classes, num_nodes]
void simple_crf_frame_set_unary(simple_crf_frame_t frame, const float* unary_energies);
void simple_crf_frame_get_unary(simple_crf_frame_t frame, float* unary_energies);

/*
 * Pairwise information getter
 */

typedef void* simple_crf_conn_iter_t;
simple_crf_conn_iter_t simple_crf_frame_pairwise_connection(simple_crf_frame_t frame, int node_i);
simple_crf_conn_iter_t simple_crf_frame_pairwise_connection_next(simple_crf_conn_iter_t iter, int *node_j);
void simple_crf_frame_pairwise_connection_end(simple_crf_conn_iter_t iter);

float simple_crf_frame_spatial_pairwise_energy(simple_crf_frame_t frame, int node_i, int node_j);
float simple_crf_frame_temporal_pairwise_energy(simple_crf_frame_t frame, simple_crf_frame_t other_frame, int node_i);


/*
 * State Getter/Setter
 */
void simple_crf_frame_get_inferred(simple_crf_frame_t frame, float* log_probas);
void simple_crf_frame_reset_inferred(simple_crf_frame_t frame);

/*
 * Inference
 */

// Returns NULL or error message
void simple_crf_inference(simple_crf_t crf, size_t max_iter);

/*
 * Utils
 */
simple_crf_t simple_crf_copy(simple_crf_t crf);

#ifdef __cplusplus
}
#endif

#endif

