#include <iostream>
#include <cstdint>
#include <cassert>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <memory>

#include "fast-slic.h"

#define CHARBIT 8

template <typename T>
static inline T my_max(T x, T y) {
    return (x > y) ? x : y;
}


template <typename T>
static inline T my_min(T x, T y) {
    return (x < y) ? x : y;
}


template <typename T>
static T fast_abs(T n)
{
    // This doesn't help much
    /*
    T const mask = n >> (sizeof(T) * CHARBIT - 1);
    return ((n + mask) ^ mask);
    */
    if (n < 0)
        return -n;
    return n;
}


extern "C" {
    void slic_assign(int H, int W, int K, uint8_t compactness_shift, uint8_t quantize_level, const uint8_t* image, const Cluster* clusters, uint32_t* assignment) {
        // Initialize
        const int16_t S = (int16_t)sqrt(H * W / K);
        std::fill_n(assignment, H * W, 0xFFFFFFFF);

        // I found threads More than 3 don't help
        #pragma omp parallel for num_threads(3)
        for (int cluster_idx = 0; cluster_idx < K; cluster_idx++) {
            const Cluster cluster = clusters[cluster_idx];

            const int16_t y_lo = my_max<int16_t>(0, cluster.y - S), y_hi = my_min<int16_t>(H, cluster.y + S);
            const int16_t x_lo = my_max<int16_t>(0, cluster.x - S), x_hi = my_min<int16_t>(W, cluster.x + S);

            for (int16_t i = y_lo; i < y_hi; i++) {
                for (int16_t j = x_lo; j < x_hi; j++) {
                    int32_t base_index = W * i + j;
                    int32_t img_base_index = 3 * base_index;

                    uint8_t r = image[img_base_index], g = image[img_base_index + 1], b = image[img_base_index + 2];

                    // OPTIMIZATION 1: floating point arithmatics is quantized down to int16_t
                    // OPTIMIZATION 2: L1 norm instead of L2
                    // OPTIMIZATION 3: L1 normalizer(x / 3) ommitted in the color distance term
                    // OPTIMIZATION 4: L1 normalizer(x / 2) ommitted in the spatial distance term
                    // OPTIMIZATION 5: assignment value is saved combined with distance and cluster number ([distance value (16 bit)] + [cluster number (16 bit)])
                    uint16_t color_dist = ((uint32_t)(fast_abs<int16_t>(r - (int16_t)cluster.r) + fast_abs<int16_t>(g - (int16_t)cluster.g) + fast_abs<int16_t>(b - (int16_t)cluster.b)) << quantize_level) >> compactness_shift;

                    uint16_t spatial_dist = ((uint32_t)(fast_abs<int16_t>(i - (int16_t)cluster.y) + fast_abs<int16_t>(j - (int16_t)cluster.x)) << quantize_level) / S; 
                    uint16_t dist = color_dist + spatial_dist; // 🙏 pray to god there was no overflow error 🙏
                    uint32_t assignment_val = ((uint32_t)dist << 16) + (uint32_t)cluster.number;

                    if (assignment[base_index] > assignment_val)
                        assignment[base_index] = assignment_val;
                }
            }
        }

        // Clean up: Drop distance part in assignment and let only cluster numbers remain
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                assignment[i * W + j] &= 0x0000FFFF; // drop the leading 2 bytes
            }
        }
    }

    void slic_update_clusters(int H, int W, int K, const uint8_t* image, Cluster* clusters, const uint32_t* assignment) {
        int num_cluster_members[K];
        int cluster_acc_vec[K][5]; // sum of [y, x, r, g, b] in cluster

        std::fill_n(num_cluster_members, K, 0);
        std::fill_n((int *)cluster_acc_vec, K * 5, 0);

        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                int base_index = W * i + j;
                int img_base_index = 3 * base_index;

                cluster_no_t cluster_no = (cluster_no_t)assignment[base_index];
                if (cluster_no == 0xFFFF) continue;
                num_cluster_members[cluster_no]++;
                cluster_acc_vec[cluster_no][0] += i;
                cluster_acc_vec[cluster_no][1] += j;
                cluster_acc_vec[cluster_no][2] += image[img_base_index];
                cluster_acc_vec[cluster_no][3] += image[img_base_index + 1];
                cluster_acc_vec[cluster_no][4] += image[img_base_index + 2];
            }
        }


        for (int k = 0; k < K; k++) {
            int num_current_members = num_cluster_members[k];
            Cluster *cluster = &clusters[k];
            cluster->num_members = num_current_members;

            if (num_current_members == 0) continue;

            // Technically speaking, as for L1 norm, you need median instead of mean for correct maximization.
            // But, I intentionally used mean here for the sake of performance.
            cluster->y = cluster_acc_vec[k][0] / num_current_members;
            cluster->x = cluster_acc_vec[k][1] / num_current_members;
            cluster->r = cluster_acc_vec[k][2] / num_current_members;
            cluster->g = cluster_acc_vec[k][3] / num_current_members;
            cluster->b = cluster_acc_vec[k][4] / num_current_members;
        }
    }

    void slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters) {
        const int S = (int)sqrt(H * W / K);

        int i = 0, j = 0;
        for (int k = 0; k < K; k++) {
            clusters[k].y = i;
            clusters[k].x = j;

            j += S;
            if (j >= W) {
                j = 0;
                i += S;
            }
            if (i >= H) {
                i = H - 1;
                j = W - 1;
            }
        }

        // TODO: move center to the position where gradient is low


        for (int k = 0; k < K; k++) {
            int base_index = W * clusters[k].y + clusters[k].x;
            int img_base_index = 3 * base_index;
            clusters[k].r = image[img_base_index];
            clusters[k].g = image[img_base_index + 1];
            clusters[k].b = image[img_base_index + 2];
            clusters[k].number = k;
            clusters[k].num_members = 0;
        }
    }

    // TODO: assign unassigned pixels
    void do_slic(int H, int W, int K, uint8_t compactness_shift, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint32_t* assignment) {
        for (int i = 0; i < max_iter; i++) {
            slic_assign(H, W, K, compactness_shift, quantize_level, image, clusters, assignment);
            slic_update_clusters(H, W, K, image, clusters, assignment);
        }
    }
}

typedef std::chrono::high_resolution_clock Clock;
int main(int argc, char** argv) {
    int K = 100;
    int compactness = 5;
    int max_iter = 2;
    int quantize_level = 7;
    try { 
        if (argc > 1) {
            K = std::stoi(std::string(argv[1]));
        }
        if (argc > 2) {
            compactness = std::stoi(std::string(argv[2]));
        }
        if (argc > 3) {
            max_iter = std::stoi(std::string(argv[3]));
        }
        if (argc > 4) {
            quantize_level = std::stoi(std::string(argv[4]));
        }
    } catch (...) {
        std::cerr << "slic num_components compactness max_iter quantize_level" << std::endl;
        return 2;
    }

    int H = 480;
    int W = 640;
    Cluster clusters[K];
    std::unique_ptr<uint8_t> image { new uint8_t[H * W * 3] };
    std::unique_ptr<uint32_t> assignment { new uint32_t[H * W] };

    std::ifstream inputf("/tmp/a.txt");
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int r, g, b;
            inputf >> r >> g >> b;
            image.get()[3 * W * i + 3 * j] = r;
            image.get()[3 * W * i + 3 * j + 1] = g;
            image.get()[3 * W * i + 3 * j + 2] = b;
        }
    }

    auto t1 = Clock::now();
    slic_initialize_clusters(H, W, K, image.get(), clusters);
    do_slic(H, W, K, compactness, quantize_level, max_iter, image.get(), clusters, assignment.get());

    auto t2 = Clock::now();
    // 6 times faster than skimage.segmentation.slic
    std::cerr << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";

    {
        std::ofstream outputf("/tmp/b.output.txt");
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                outputf << (short)assignment.get()[W * i + j] << " ";
            }
            outputf << std::endl;
        }
    }
    {
        std::ofstream outputf("/tmp/b.clusters.txt");
        for (int k = 0; k < K; k++) {
            outputf << clusters[k].y << " " << clusters[k].x << " " << clusters[k].num_members << std::endl;
        }
    }
    return 0;
}
