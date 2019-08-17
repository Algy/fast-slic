#include <iostream>
#include <algorithm>
#include <cmath>
#include "lsc.h"

//map pixels into ten dimensional feature space

namespace fslic {
    void ContextLSC::before_iteration() {
        map_image_into_feature_space();
        map_centroids_into_feature_space();
    }

    void ContextLSC::map_image_into_feature_space() {
        const float PI = 3.1415926;
        const float halfPI = PI / 2;
        const float ratio = compactness / 100.0f;

        int len = H * W;

        for (auto &plane : image_planes) plane.resize(len);
        for (auto &feat : image_features) feat.resize(len);
        image_weights.resize(len);
        for (int i = 0; i < H; i++) {
            const uint8_t* image_row = &aligned_quad_image[quad_image_memory_width * i];
            for (int j = 0; j < W; j++) {
                int index = i * W + j;
                image_planes[0][index] = image_row[4 * j];
                image_planes[1][index] = image_row[4 * j + 1];
                image_planes[2][index] = image_row[4 * j + 2];
            }
        }

        {
            // l1, l2, a1, a2, b1, b2
            float* __restrict feat_1 = &image_features[0][0];
            float* __restrict feat_2 = &image_features[1][0];
            float* __restrict feat_3 = &image_features[2][0];
            float* __restrict feat_4 = &image_features[3][0];
            float* __restrict feat_5 = &image_features[4][0];
            float* __restrict feat_6 = &image_features[5][0];

            const uint8_t* __restrict L = &image_planes[0][0];
            const uint8_t* __restrict A = &image_planes[1][0];
            const uint8_t* __restrict B = &image_planes[2][0];
            for (int i = 0; i < len; i++) {
                float theta_L = halfPI * (L[i] / 255.0f);
                float theta_A = halfPI * (A[i] / 255.0f);
                float theta_B = halfPI * (B[i] / 255.0f);

                feat_1[i] = cos(theta_L) * 2.55f;
    			feat_2[i] = sin(theta_L) * 2.55f;
    			feat_3[i] = cos(theta_A) * 2.55f;
    			feat_4[i] = sin(theta_A) * 2.55f;
    			feat_5[i] = cos(theta_B) * 2.55f;
    			feat_6[i] = sin(theta_B) * 2.55f;
            }
        }
        {
            // x1, x2, y1, y2
            float* __restrict feat_1 = &image_features[6][0];
            float* __restrict feat_2 = &image_features[7][0];
            float* __restrict feat_3 = &image_features[8][0];
            float* __restrict feat_4 = &image_features[9][0];
            for (int i = 0; i < len; i++) {
                float y = i / W, x = i % W;
        		float theta_x = halfPI * (x / S);
        		float theta_y = halfPI * (y / S);
    			feat_1[i] = ratio * cos(theta_x);
    			feat_2[i] = ratio * sin(theta_x);
    			feat_3[i] = ratio * cos(theta_y);
    			feat_4[i] = ratio * sin(theta_y);
            }
        }

	    float sum_features[10];
        std::fill_n(sum_features, 10, 0);
        {
            for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                float* __restrict feat = &image_features[ix_feat][0];
                float sum = 0;
                for (int i = 0; i < len; i++) {
                    sum += feat[i];
                }
                sum_features[ix_feat] = sum / len;
            }
        }

        {
            float* __restrict Weight = &image_weights[0];
            for (int i = 0; i < len; i++) {
                float w = 0;
                for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                    w += sum_features[ix_feat] * image_features[ix_feat][i];
                }
                Weight[i] = w;
            }

            for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                float* __restrict feat = &image_features[ix_feat][0];
                for (int i = 0; i < len; i++) {
                    feat[i] /= Weight[i];
                }
            }
        }
    }

    void ContextLSC::map_centroids_into_feature_space() {
        for (std::vector<float> &feat : centroid_features) {
            feat.resize(K);
            std::fill(feat.begin(), feat.end(), 0);
        }

        for (int k = 0; k < K; k++) {
            const Cluster* cluster = &clusters[k];
            if (cluster->num_members <= 0) continue;
            int index = clamp<int>(cluster->y, 0, H - 1) * W + clamp<int>(cluster->x, 0, W - 1);
            for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                centroid_features[ix_feat][k] = image_features[ix_feat][index];
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

            int y_lo = my_max<int>(cluster_y - S, 0), y_hi = my_min<int>(cluster_y + S, H - 1);
            int x_lo = my_max<int>(cluster_x - S, 0), x_hi = my_min<int>(cluster_x + S, W - 1);

            for (int i = y_lo; i <= y_hi; i++) {
                if (!valid_subsample_row(i)) continue;
                for (int j = x_lo; j <= x_hi; j++) {
                    float &min_dist = aligned_min_dists[min_dist_memory_width * i + j];
                    uint16_t &assignment = aligned_assignment[assignment_memory_width * i + j];
                    int index = W * i + j;
                    float dist = 0;
                    for (int ix_feat = 0; ix_feat < 10; ix_feat++) {
                        float diff = img_feats[ix_feat][index] - centroid_feats[ix_feat][cluster_no];
                        dist += diff * diff;
                    }
                    if (min_dist > dist) {
                        min_dist = dist;
                        assignment = cluster_no;
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
                    uint16_t cluster_no = aligned_assignment[assignment_memory_width * i + j];
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
}
