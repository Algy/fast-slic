#include <iostream>
#include <algorithm>
#include <cmath>
#include "lsc.h"

//map pixels into ten dimensional feature space

namespace fslic {
    void ContextLSC::before_iteration() {
        map_image_into_feature_space();
        #ifdef FAST_SLIC_TIMER
        auto t1 = Clock::now();
        #endif
        map_centroids_into_feature_space();
        #ifdef FAST_SLIC_TIMER
        auto t2 = Clock::now();
        std::cerr << "LSC.centroid: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us\n";
        #endif
    }

    ContextLSC::~ContextLSC() {
        if (uint8_memory_pool) delete [] uint8_memory_pool;
        if (float_memory_pool) delete [] float_memory_pool;
    }

    void ContextLSC::map_image_into_feature_space() {
        const float PI = 3.1415926;
        const float halfPI = PI / 2;
        const float ratio = compactness / 100.0f;

        int len = H * W;
        int aligned_len = simd_helper::align_to_next(len);

        #ifdef FAST_SLIC_TIMER
        auto t0 = Clock::now();
        #endif
        if (!uint8_memory_pool) uint8_memory_pool = new uint8_t[3 * len];
        if (!float_memory_pool) float_memory_pool = new float[11 * aligned_len];
        image_planes[0] = &uint8_memory_pool[0];
        image_planes[1] = &uint8_memory_pool[len];
        image_planes[2] = &uint8_memory_pool[2 * len];
        for (int i = 0; i < 10; i++) {
            image_features[i] = &float_memory_pool[i * aligned_len];
        }
        image_weights = &float_memory_pool[10 * aligned_len];

        #ifdef FAST_SLIC_TIMER
        auto t1 = Clock::now();
        #endif
        #pragma omp parallel for
        for (int i = 0; i < H; i++) {
            const uint8_t* image_row = quad_image.get_row(i);
            for (int j = 0; j < W; j++) {
                int index = i * W + j;
                image_planes[0][index] = image_row[4 * j];
                image_planes[1][index] = image_row[4 * j + 1];
                image_planes[2][index] = image_row[4 * j + 2];
            }
        }

        float* __restrict img_feats[10];
        for (int i = 0; i < 10; i++) {
            img_feats[i] = &image_features[i][0];
        }

        #ifdef FAST_SLIC_TIMER
        auto t2 = Clock::now();
        #endif
        {
            // l1, l2, a1, a2, b1, b2
            float color_sine_map[256];
            float color_cosine_map[256];
            int spatial_max = my_max(H, W);
            std::vector<float> spatial_cosine_map(spatial_max);
            std::vector<float> spatial_sine_map(spatial_max);
            for (int X = 0; X < 256; X++) {
                float theta = halfPI * (X / 255.0f);
                color_cosine_map[X] = cos(theta) * 2.55f;
    			color_sine_map[X] = sin(theta) * 2.55f;
            }
            for (int i = 0; i < spatial_max; i++) {
                float theta = i * (halfPI / S);
                spatial_cosine_map[i] = ratio * cos(theta);
                spatial_sine_map[i] = ratio * sin(theta);
            }

            const uint8_t* __restrict L = &image_planes[0][0];
            const uint8_t* __restrict A = &image_planes[1][0];
            const uint8_t* __restrict B = &image_planes[2][0];
            #pragma omp parallel for
            for (int i = 0; i < len; i++) {
                img_feats[0][i] = color_cosine_map[L[i]];
    			img_feats[1][i] = color_sine_map[L[i]];
    			img_feats[2][i] = color_cosine_map[A[i]];
    			img_feats[3][i] = color_sine_map[A[i]];
    			img_feats[4][i] = color_cosine_map[B[i]];
    			img_feats[5][i] = color_sine_map[B[i]];
            }
            // x1, x2, y1, y2

            #pragma omp parallel for
            for (int y = 0; y < H; y++) {
                std::copy(
                    spatial_cosine_map.begin(),
                    spatial_cosine_map.begin() + W,
                    &img_feats[6][y * W]
                );
                std::copy(
                    spatial_sine_map.begin(),
                    spatial_sine_map.begin() + W,
                    &img_feats[7][y * W]
                );
            }

            #pragma omp parallel for
            for (int y = 0; y < H; y++) {
                std::fill_n(&img_feats[8][y * W], W, spatial_cosine_map[y]);
                std::fill_n(&img_feats[9][y * W], W, spatial_sine_map[y]);
            }
        }
        #ifdef FAST_SLIC_TIMER
        auto t3 = Clock::now();
        #endif

	    float sum_features[10];
        std::fill_n(sum_features, 10, 0);
        {
            #pragma omp parallel for
            for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                float sum = 0;
                for (int i = 0; i < len; i++) {
                    sum += img_feats[ix_feat][i];
                }
                sum_features[ix_feat] = sum / len;
            }
        }

        #ifdef FAST_SLIC_TIMER
        auto t4 = Clock::now();
        #endif

        {
            #pragma omp parallel for
            for (int i = 0; i < len; i++) {
                float w = 0;
                for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                    w += sum_features[ix_feat] * img_feats[ix_feat][i];
                }
                image_weights[i] = w;
            }
            normalize_features(len);
        }
        #ifdef FAST_SLIC_TIMER
        auto t5 = Clock::now();
        std::cerr << "LSC.image_alloc: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << "us\n";
        std::cerr << "LSC.image_copy: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us\n";
        std::cerr << "LSC.feature_map: " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << "us\n";
        std::cerr << "LSC.weight_map: " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << "us\n";
        std::cerr << "LSC.feature_normalize: " << std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count() << "us\n";
        #endif
    }

    void ContextLSC::map_centroids_into_feature_space() {
        for (std::vector<float> &feat : centroid_features) {
            feat.resize(K);
            std::fill(feat.begin(), feat.end(), 0);
        }

        const float* __restrict img_feats[10];
        float* __restrict centroid_feats[10];

        for (int i = 0; i < 10; i++) {
            img_feats[i] = &image_features[i][0];
            centroid_feats[i] = &centroid_features[i][0];
        }

        #pragma omp parallel for
        for (int k = 0; k < K; k++) {
            const Cluster* cluster = &clusters[k];
            if (cluster->num_members <= 0) continue;
            int index = clamp<int>(cluster->y, 0, H - 1) * W + clamp<int>(cluster->x, 0, W - 1);
            for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                centroid_feats[ix_feat][k] = img_feats[ix_feat][index];
            }
        }
    }

    void ContextLSC::assign_clusters(const Cluster** target_clusters, int size) {
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

            for (int i = y_lo; i < y_hi; i++) {
                if (!valid_subsample_row(i)) continue;
                for (int j = x_lo; j < x_hi; j++) {
                    float &min_dist = min_dists.get(i, j);
                    uint16_t &label = assignment.get(i, j);
                    int index = W * i + j;
                    float dist = 0;
                    for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                        float diff = img_feats[ix_feat][index] - centroid_feats[ix_feat][cluster_no];
                        dist += diff * diff;
                    }
                    if (min_dist > dist) {
                        min_dist = dist;
                        label = cluster_no;
                    }
                }
            }
        }
    }

    void ContextLSC::after_update() {
        const float* __restrict img_feats[10];
        float* __restrict centroid_feats[10];

        float* __restrict wsums[10];
        for (int i = 0; i < 10; i++) {
            img_feats[i] = &image_features[i][0];
            centroid_feats[i] = &centroid_features[i][0];
            std::fill_n(centroid_feats[i], K, 0.0f);
            wsums[i] = new float[K];
            std::fill_n(wsums[i], K, 0.0f);
        }

        #pragma omp parallel
        {
            float* __restrict local_feats[10];
            float* __restrict local_wsums[10];
            for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                local_feats[ix_feat] = new float[K];
                local_wsums[ix_feat] = new float[K];
                std::fill_n(local_feats[ix_feat], K, 0);
                std::fill_n(local_wsums[ix_feat], K, 0);
            }

            #pragma omp for
            for (int i = fit_to_stride(0); i < H; i += subsample_stride) {
                for (int j = 0; j < W; j++) {
                    uint16_t cluster_no = assignment.get(i, j);
                    if (cluster_no == 0xFFFF) continue;
                    int index = W * i + j;
                    float w = image_weights[index];
                    for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                        local_feats[ix_feat][cluster_no] += w * img_feats[ix_feat][index];
                        local_wsums[ix_feat][cluster_no] += w;
                    }
                }
            }

            #pragma omp critical
            {
                for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                    for (int k = 0; k < K; k++) {
                        centroid_feats[ix_feat][k] += local_feats[ix_feat][k];
                        wsums[ix_feat][k] += local_wsums[ix_feat][k];
                    }
                }
            }

            for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                delete [] local_feats[ix_feat];
                delete [] local_wsums[ix_feat];
            }
        }

        for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
            for (int k = 0; k < K; k++) {
                centroid_feats[ix_feat][k] /= wsums[ix_feat][k];
            }
        }

        for (int i = 0; i < 10; i++) {
            delete [] wsums[i];
        }
    }

	void ContextLSC::normalize_features(int size) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                image_features[ix_feat][i] /= image_weights[i];
            }
        }
    }
}
