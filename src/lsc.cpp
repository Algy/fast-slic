#include <iostream>
#include <algorithm>
#include <cmath>
#include "lsc.h"
#include "cielab.h"
#include "parallel.h"
#include "timer.h"

//map pixels into ten dimensional feature space

namespace fslic {
    void ContextLSC::before_iteration() {
        map_image_into_feature_space();
        map_centroids_into_feature_space();
    }

    ContextLSC::~ContextLSC() {
        if (uint8_memory_pool) delete [] uint8_memory_pool;
        if (float_memory_pool) delete [] float_memory_pool;
    }

    void ContextLSC::map_image_into_feature_space() {
        fstimer::Scope s("map_image_into_feature_space");

        const float PI = 3.1415926;
        const float halfPI = PI / 2;
        const float ratio = compactness / 100.0f;
        const float C_spatial = C_color * ratio;

        int len = H * W;
        int aligned_len = simd_helper::align_to_next(len);
        int aligned_K = simd_helper::align_to_next(K);

        {
            fstimer::Scope s("image_alloc");

            if (uint8_memory_pool) delete [] uint8_memory_pool;
            uint8_memory_pool = new uint8_t[3 * aligned_len];
            if (float_memory_pool) delete [] uint8_memory_pool;
            float_memory_pool = new float[11 * aligned_len + 10 * aligned_K];

            image_planes[0] = &uint8_memory_pool[0];
            image_planes[1] = &uint8_memory_pool[aligned_len];
            image_planes[2] = &uint8_memory_pool[2 * aligned_len];
            for (int i = 0; i < 10; i++) {
                image_features[i] = &float_memory_pool[i * aligned_len];
                centroid_features[i] = &float_memory_pool[11 * aligned_len + i * aligned_K];
            }
            image_weights = &float_memory_pool[10 * aligned_len];
        }

        {
            fstimer::Scope s("image_copy");
            #pragma omp parallel for num_threads(fsparallel::nth())
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    int index = i * W + j;
                    image_planes[0][index] = image[3 * index];
                    image_planes[1][index] = image[3 * index + 1];
                    image_planes[2][index] = image[3 * index + 2];
                }
            }
        }

        {
            fstimer::Scope s("feature_map");

            // l1, l2, a1, a2, b1, b2
            float color_sine_map[256];
            float color_cosine_map[256];
            float L_sine_map[256];
            float L_cosine_map[256];
            std::vector<float> width_cosine_map(W);
            std::vector<float> width_sine_map(W);
            std::vector<float> height_cosine_map(H);
            std::vector<float> height_sine_map(H);
            for (int X = 0; X < 256; X++) {
                float theta = halfPI * (X / 255.0f);
                float cosine = cos(theta), sine = sin(theta);
                color_cosine_map[X] = C_color * cosine * 2.55f;
    			color_sine_map[X] = C_color * sine * 2.55f;
            }

            for (int X = 0; X < 256; X++) {
                float theta = halfPI * (X / 255.0f);
                L_cosine_map[X] = C_color * cos(theta);
                L_sine_map[X] = C_color * sin(theta);
            }

            for (int i = 0; i < H; i++) {
                float theta = i * (halfPI / S);
                height_cosine_map[i] = C_spatial * cos(theta);
                height_sine_map[i] = C_spatial * sin(theta);
            }

            for (int i = 0; i < W; i++) {
                float theta = i * (halfPI / S);
                width_cosine_map[i] = C_spatial * cos(theta);
                width_sine_map[i] = C_spatial * sin(theta);
            }

            const uint8_t* __restrict L = &image_planes[0][0];
            const uint8_t* __restrict A = &image_planes[1][0];
            const uint8_t* __restrict B = &image_planes[2][0];
            #pragma omp parallel for num_threads(fsparallel::nth())
            for (int i = 0; i < len; i++) {
                image_features[0][i] = L_cosine_map[L[i]];
    			image_features[1][i] = L_sine_map[L[i]];
    			image_features[2][i] = color_cosine_map[A[i]];
    			image_features[3][i] = color_sine_map[A[i]];
    			image_features[4][i] = color_cosine_map[B[i]];
    			image_features[5][i] = color_sine_map[B[i]];
            }
            // x1, x2, y1, y2

            #pragma omp parallel for num_threads(fsparallel::nth())
            for (int y = 0; y < H; y++) {
                std::copy(
                    width_cosine_map.begin(),
                    width_cosine_map.end(),
                    &image_features[6][y * W]
                );
                std::copy(
                    width_sine_map.begin(),
                    width_sine_map.end(),
                    &image_features[7][y * W]
                );
            }

            #pragma omp parallel for num_threads(fsparallel::nth())
            for (int y = 0; y < H; y++) {
                std::fill_n(&image_features[8][y * W], W, height_cosine_map[y]);
                std::fill_n(&image_features[9][y * W], W, height_sine_map[y]);
            }
        }

	    float sum_features[10];
        {
            fstimer::Scope s("weight_map");
            std::fill_n(sum_features, 10, 0);
            #pragma omp parallel for num_threads(fsparallel::nth())
            for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                float sum = 0;
                for (int i = 0; i < len; i++) {
                    sum += image_features[ix_feat][i];
                }
                sum_features[ix_feat] = sum / len;
            }
        }
        {
            fstimer::Scope s("normalize_features");
            #pragma omp parallel for num_threads(fsparallel::nth())
            for (int i = 0; i < len; i++) {
                float w = 0;
                for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                    w += sum_features[ix_feat] * image_features[ix_feat][i];
                }
                image_weights[i] = w;
            }
            normalize_features(image_features, image_weights, len);
        }
    }

    void ContextLSC::map_centroids_into_feature_space() {
        fstimer::Scope s("map_centroids_into_feature_space");

        float* __restrict wsums = new float[K];
        std::fill_n(wsums, K, 0.0f);

        for (float *feat : centroid_features) {
            std::fill_n(feat, K, 0);
        }

        #pragma omp parallel for num_threads(fsparallel::nth())
        for (int k = 0; k < K; k++) {
            const Cluster* cluster = &clusters[k];
            int cluster_y = cluster->y, cluster_x = cluster->x;
            int y_lo = my_max<int>(cluster_y - S / 4, 0), y_hi = my_min<int>(cluster_y + S / 4 + 1, H);
            int x_lo = my_max<int>(cluster_x - S / 4, 0), x_hi = my_min<int>(cluster_x + S / 4 + 1, W);

            for (int i = y_lo; i < y_hi; i++) {
                for (int j = x_lo; j < x_hi; j++) {
                    int index = W * i + j;
                    // float weight = image_weights[index];
                    for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                        centroid_features[ix_feat][k] += image_features[ix_feat][index];
                    }
                    wsums[k] += 1.0f;
                }
            }
        }
        normalize_features(centroid_features, wsums, K);
        delete [] wsums;
    }

    void ContextLSC::after_update() {
    }

	void ContextLSC::normalize_features(float *__restrict numers[10], float* __restrict weights, int size) {
        #pragma omp parallel for num_threads(fsparallel::nth())
        for (int i = 0; i < size; i++) {
            for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                numers[ix_feat][i] /= weights[i];
            }
        }
    }
}
