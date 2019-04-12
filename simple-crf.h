#ifndef _SIMPLE_CRF_H
#define _SIMPLE_CRF_H

#include <stdint.h>

struct SimpleCRF;
typedef struct SimpleCRFConnectivity {
    int time;
    int num_nodes;
    int *num_neighbors;
    int **neighbors;
} SimpleCRFConnectivity;

typedef struct SimpleCRFParams {
    float spatial_w;
    float temporal_w;
    float spatial_srgb;
    float temporal_srgb;
} SimpleCRFParams;


typedef struct SimpleCRF* simple_crf_t;

#ifdef __cplusplus
extern "C" {
#endif
simple_crf_t simple_crf_new(int num_classes, int num_nodes);
void simple_crf_free(simple_crf_t crf);

SimpleCRFParams simple_crf_get_params(simple_crf_t crf);
void simple_crf_set_params(simple_crf_t crf, SimpleCRFParams params);

int simple_crf_first_time(simple_crf_t crf);
int simple_crf_last_time(simple_crf_t crf);
int simple_crf_num_time_frames(simple_crf_t crf);
int simple_crf_pop_time_frame(simple_crf_t crf);
int simple_crf_push_time_frame(simple_crf_t crf);


void simple_crf_set_pixels(simple_crf_t crf, int t, const uint8_t* yxrgbs);
void simple_crf_set_connectivity(simple_crf_t crf, int t, const SimpleCRFConnectivity* conn);
// classes: int[] of shape [num_nodes]
void simple_crf_set_frame_mask(simple_crf_t crf, int t, const int* classes, float confidence);
// probas: float[] of shape [num_classes, num_nodes]
void simple_crf_set_frame_log_proba(simple_crf_t crf, int t, const float* probas);
void simple_crf_set_frame_unbiased(simple_crf_t crf, int t);
void simple_crf_reset_frame_state(simple_crf_t crf);

const char* simple_crf_iterate(simple_crf_t crf, int max_iter);

// out_probas: flaot[] of shape [num_classes, num_nodes]
void simple_crf_get_log_proba(simple_crf_t crf, int t, float* out_log_probas); 
simple_crf_t simple_crf_copy(simple_crf_t crf);


#ifdef __cplusplus
}
#endif

#endif

