#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <cstring>

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

template <typename T>
static T ceil_int(T numer, T denom) {
    return (numer + denom - 1) / denom;
}

struct ClusterPixel {
    uint16_t cluster_nos[9];
    int8_t last_index;
};


struct Context {
    const uint8_t* image;
    const char* algorithm;
    int H;
    int W;
    int K;
    uint8_t compactness_shift;
    uint8_t quantize_level;
    Cluster* clusters;
    uint32_t* assignment;
    ClusterPixel *cluster_boxes;
    ClusterPixel *cluster_pixels;
};

static void init_context(Context *context) {
    memset(context, 0, sizeof(Context));
}

static void free_context(Context *context) {
    if (context->cluster_boxes)
        delete [] context->cluster_boxes;
    if (context->cluster_pixels)
        delete [] context->cluster_pixels;
}

static void slic_assign_cluster_oriented(Context *context) {
    auto H = context->H;
    auto W = context->W;
    auto K = context->K;
    auto compactness_shift = context->compactness_shift;
    auto clusters = context->clusters;
    auto image = context->image;
    auto assignment = context->assignment;
    auto quantize_level = context->quantize_level;

    const int16_t S = (int16_t)sqrt(H * W / K);
    std::fill_n(assignment, H * W, 0xFFFFFFFF);

    uint8_t spatial_shift = quantize_level + compactness_shift;
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
                uint16_t color_dist = ((uint32_t)(fast_abs<int16_t>(r - (int16_t)cluster.r) + fast_abs<int16_t>(g - (int16_t)cluster.g) + fast_abs<int16_t>(b - (int16_t)cluster.b)) << quantize_level);

                uint16_t spatial_dist = ((uint32_t)(fast_abs<int16_t>(i - (int16_t)cluster.y) + fast_abs<int16_t>(j - (int16_t)cluster.x)) << spatial_shift) / S; 
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

static inline uint32_t get_assignment_val(int16_t S, int i, int j, uint8_t r, uint8_t g, uint8_t b, const Cluster* cluster, uint8_t quantize_level, uint8_t spatial_shift) {
    uint16_t color_dist = ((uint32_t)(fast_abs<int16_t>(r - (int16_t)cluster->r) + fast_abs<int16_t>(g - (int16_t)cluster->g) + fast_abs<int16_t>(b - (int16_t)cluster->b)) << quantize_level);

    uint16_t spatial_dist = ((uint32_t)(fast_abs<int16_t>(i - (int16_t)cluster->y) + fast_abs<int16_t>(j - (int16_t)cluster->x)) << spatial_shift) / S; 
    uint16_t dist = color_dist + spatial_dist;
    return ((uint32_t)dist << 16) + (uint32_t)cluster->number;
}

static inline uint32_t get_assignment_box_val(const ClusterPixel *box, int16_t S, int i, int j, uint8_t r, uint8_t g, uint8_t b, const Cluster* clusters, uint8_t quantize_level, uint8_t spatial_shift) {
    uint32_t min_val = 0xFFFFFFFF;
    for (int8_t k = 0; k <= box->last_index; k++) {
        uint16_t cluster_no = box->cluster_nos[k];
        const Cluster *cluster = &clusters[cluster_no];
        if (cluster->y - S <= i && i < cluster->y + S && cluster->x - S <= j && j < cluster->x + S) {
            uint32_t assignment_val = get_assignment_val(S, i, j, r, g, b, cluster, quantize_level, spatial_shift);
            if (min_val > assignment_val)
                min_val = assignment_val;
        }
    }
    return min_val;
}

#include <iostream>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;
static void slic_assign_pixel_oriented(Context* context) {
    auto H = context->H;
    auto W = context->W;
    auto K = context->K;
    auto compactness_shift = context->compactness_shift;
    auto clusters = context->clusters;
    auto image = context->image;
    auto assignment = context->assignment;
    auto quantize_level = context->quantize_level;

    const int16_t S = (int16_t)sqrt(H * W / K);

    // left, right, top, bottom borders are included
    int box_H = ceil_int(H, (int)S) + 2;
    int box_W = ceil_int(W, (int)S) + 2;

    if (!context->cluster_boxes) {
        context->cluster_boxes = new ClusterPixel[box_H * box_W];
    }

    ClusterPixel* cluster_boxes = context->cluster_boxes;
    for (int i = 0; i < box_H; i++) {
        for (int j = 0; j < box_W; j++) {
            cluster_boxes[i * box_W + j].last_index = -1;
        }
    }

    for (int cluster_idx = 0; cluster_idx < K; cluster_idx++) {
        int32_t base_index = box_W * (clusters[cluster_idx].y / S + 1) + (clusters[cluster_idx].x / S + 1);
        int8_t last_index = cluster_boxes[base_index].last_index;
        if (last_index >= 8) continue;
        cluster_boxes[base_index].cluster_nos[last_index + 1] = cluster_idx;
        cluster_boxes[base_index].last_index = last_index + 1;
    }

    auto t1 = Clock::now();

    uint8_t spatial_shift = quantize_level + compactness_shift;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int32_t base_index = W * i + j;
            int32_t img_base_index = 3 * base_index;

            uint8_t r = image[img_base_index], g = image[img_base_index + 1], b = image[img_base_index + 2];

            int center_box_i = i / S + 1;
            int center_box_j = j / S + 1;

            uint32_t min_val = 0xFFFFFFFF;

            #pragma GCC unroll(9)
            for (int8_t delta = 0; delta < 9; delta++) {
                int di = delta / 3 - 1, dj = delta % 3 - 1;
                // [di, dj] = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
                //
                int box_i = center_box_i + di, box_j = center_box_j + dj;
                const ClusterPixel *box = &cluster_boxes[box_i * box_W + box_j];
                uint32_t assignment_val = get_assignment_box_val(box,  S, i, j, r, g, b, clusters, quantize_level, spatial_shift);
                if (min_val > assignment_val)
                    min_val = assignment_val;
            }

            // Drop distance part in assignment and let only cluster numbers remain
            assignment[base_index] = min_val & 0x0000FFFF;
        }
    }

    auto t2 = Clock::now();

    std::cerr << "ASS " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
}

static void slic_assign(Context *context) {
    auto t1 = Clock::now();
    if (!strcmp(context->algorithm, "cluster_oriented")) {
        slic_assign_cluster_oriented(context);
    } else if (!strcmp(context->algorithm, "pixel_oriented")) {
        slic_assign_pixel_oriented(context);
    }
    auto t2 = Clock::now();
    std::cerr << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
}

static void slic_update_clusters(Context *context) {
    auto H = context->H;
    auto W = context->W;
    auto K = context->K;
    auto image = context->image;
    auto clusters = context->clusters;
    auto assignment = context->assignment;

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


extern "C" {

    void slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters) {
        const int S = (int)sqrt(H * W / K);

        int *gradients = new int[H * W];

        // compute gradients
        std::fill_n(gradients, H * W, 1 << 21);
        for (int i = 1; i < H; i += S) {
            for (int j = 1; j < W; j += S) {
                int base_index = i * W + j;
                int img_base_index = 3 * base_index;
                int dx = 
                    fast_abs(image[img_base_index + 3] - image[img_base_index - 3]) +
                    fast_abs(image[img_base_index + 4] - image[img_base_index - 2]) +
                    fast_abs(image[img_base_index + 5] - image[img_base_index - 1]);
                int dy = 
                    fast_abs(image[img_base_index + 3 * W] - image[img_base_index - 3 * W]) +
                    fast_abs(image[img_base_index + 3 * W + 1] - image[img_base_index - 3 * W + 1]) +
                    fast_abs(image[img_base_index + 3 * W + 2] - image[img_base_index - 3 * W + 2]);
                gradients[base_index] = dx + dy;
            }
        }

        int acc_k = 0;
        for (int i = 0; i < H; i += S) {
            for (int j = 0; j < W; j += S) {
                if (acc_k >= K) break;

                int eh = my_min<int>(i + S, H - 1), ew = my_min<int>(j + S, W - 1);
                int center_y = i + S / 2, center_x = j + S / 2;
                int min_gradient = 1 << 21;
                for (int ty = i; ty < eh; ty++) {
                    for (int tx = j; tx < ew; tx++) {
                        int base_index = ty * W + tx;
                        if (min_gradient > gradients[base_index]) {
                            center_y = ty;
                            center_x = tx;
                            min_gradient = gradients[base_index];
                        }

                    }
                }

                clusters[acc_k].y = center_y;
                clusters[acc_k].x = center_x;


                acc_k++;
            }
        }

        while (acc_k < K) {
            clusters[acc_k].y = H / 2;
            clusters[acc_k].x = W / 2;
            acc_k++;
        }

        delete [] gradients;


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

    static void slic_enforce_connectivity(int H, int W, int K, const Cluster* clusters, uint32_t* assignment) {
        if (K <= 0) return;

        uint8_t *visited = new uint8_t[H * W];
        std::fill_n(visited, H * W, 0);

        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                int base_index = W * i + j;
                if (assignment[base_index] != 0xFFFF) continue;

                std::vector<int> visited_indices;
                std::vector<int> stack;
                std::unordered_set<int> adj_cluster_indices;
                stack.push_back(base_index);
                while (!stack.empty()) {
                    int index = stack.back();
                    stack.pop_back();

                    if (assignment[index] != 0xFFFF) {
                        adj_cluster_indices.insert(assignment[index]);
                        continue;
                    } else if (visited[index]) {
                        continue;
                    }
                    visited[index] = 1;
                    visited_indices.push_back(index);

                    int index_j = index % W;
                    // up
                    if (index > W) {
                        stack.push_back(index - W);
                    }

                    // down
                    if (index + W < H * W) {
                        stack.push_back(index + W);
                    }

                    // left
                    if (index_j > 0) {
                        stack.push_back(index - 1);
                    }

                    // right
                    if (index_j + 1 < W) {
                        stack.push_back(index + 1);
                    }
                }

                int target_cluster_index = 0;
                int max_num_members = 0;
                for (auto it = adj_cluster_indices.begin(); it != adj_cluster_indices.end(); ++it) {
                    const Cluster* adj_cluster = &clusters[*it];
                    if (max_num_members < adj_cluster->num_members) {
                        target_cluster_index = adj_cluster->number;
                        max_num_members = adj_cluster->num_members;
                    }
                }

                for (auto it = visited_indices.begin(); it != visited_indices.end(); ++it) {
                    assignment[*it] = target_cluster_index;
                }

            }
        }
        delete [] visited;
    }

    void do_slic(int H, int W, int K, uint8_t compactness_shift, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint32_t* assignment) {

        Context context;
        init_context(&context);
        context.image = image;
        context.algorithm = "pixel_oriented";
        context.H = H;
        context.W = W;
        context.K = K;
        context.compactness_shift = compactness_shift;
        context.quantize_level = quantize_level;
        context.clusters = clusters;
        context.assignment = assignment;

        for (int i = 0; i < max_iter; i++) {
            slic_assign(&context);
            slic_update_clusters(&context);
        }
        slic_enforce_connectivity(H, W, K, clusters, assignment);

        free_context(&context);
    }
}

#ifdef PROTOTYPE_MAIN_DEMO
#include <string>
#include <chrono>
#include <fstream>
#include <memory>
#include <iostream>
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
    std::unique_ptr<uint8_t[]> image { new uint8_t[H * W * 3] };
    std::unique_ptr<uint32_t[]> assignment { new uint32_t[H * W] };

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
#endif
