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

template <typename T>
static T round_int(T numer, T denom) {
    return (numer + (denom / 2)) / denom;
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
    int16_t S;
    uint8_t compactness;
    uint8_t quantize_level;
    Cluster* clusters;
    uint32_t* assignment;
    ClusterPixel *cluster_boxes;
    uint16_t* spatial_normalize_cache; // (x) -> (uint16_t)(((uint32_t)x << quantize_level) * M / S / 2 * 3) 
};

static void init_context(Context *context) {
    memset(context, 0, sizeof(Context));
}

static void free_context(Context *context) {
    if (context->cluster_boxes)
        delete [] context->cluster_boxes;
    if (context->spatial_normalize_cache)
        delete [] context->spatial_normalize_cache;
}

uint32_t calc_z_order(uint16_t yPos, uint16_t xPos)
{
    static const uint32_t MASKS[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
    static const uint32_t SHIFTS[] = {1, 2, 4, 8};

    uint32_t x = xPos;  // Interleave lower 16 bits of x and y, so the bits of x
    uint32_t y = yPos;  // are in the even positions and bits from y in the odd;

    x = (x | (x << SHIFTS[3])) & MASKS[3];
    x = (x | (x << SHIFTS[2])) & MASKS[2];
    x = (x | (x << SHIFTS[1])) & MASKS[1];
    x = (x | (x << SHIFTS[0])) & MASKS[0];

    y = (y | (y << SHIFTS[3])) & MASKS[3];
    y = (y | (y << SHIFTS[2])) & MASKS[2];
    y = (y | (y << SHIFTS[1])) & MASKS[1];
    y = (y | (y << SHIFTS[0])) & MASKS[0];

    const uint32_t result = x | (y << 1);
    return result;
}


static uint64_t get_sort_value(int16_t y, int16_t x, int16_t S) {
    // return ((uint64_t)(y / (2 * S)) << 48) + ((uint64_t)(x / (2 * S)) << 32) + (uint32_t)calc_z_order(y, x);
    return calc_z_order(y, x);
    // return y + x;
}


struct sort_cmp {
    int16_t S;
    sort_cmp(int16_t S) : S(S) {};
    inline bool operator() (const Cluster * lhs, const Cluster * rhs) {
        return get_sort_value(lhs->y, lhs->x, S) < get_sort_value(rhs->y, rhs->x, S);
    }
};


#include <string>
#include <chrono>
#include <fstream>
#include <memory>
#include <iostream>
typedef std::chrono::high_resolution_clock Clock;

static inline uint32_t get_assignment_value(const Cluster* cluster, const uint8_t* image, int32_t base_index, uint16_t spatial_dist, uint8_t quantize_level) {
    int32_t img_base_index = 3 * base_index;
    uint8_t r = image[img_base_index], g = image[img_base_index + 1], b = image[img_base_index + 2];
    uint16_t color_dist = ((uint32_t)(fast_abs<int16_t>(r - (int16_t)cluster->r) + fast_abs<int16_t>(g - (int16_t)cluster->g) + fast_abs<int16_t>(b - (int16_t)cluster->b)) << quantize_level);
    return ((uint32_t)(color_dist + spatial_dist) << 16) + (uint32_t)cluster->number;
}

static void slic_assign_cluster_oriented(Context *context) {
    auto H = context->H;
    auto W = context->W;
    auto K = context->K;
    auto compactness = context->compactness;
    auto clusters = context->clusters;
    auto image = context->image;
    auto assignment = context->assignment;
    auto quantize_level = context->quantize_level;

    const int16_t S = context->S;

    #if _OPENMP >= 200805
    #pragma omp parallel for collapse(2)
    #else
    #pragma omp parallel for
    #endif
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            assignment[i * W + j] =  0xFFFFFFFF;
        }
    }

    if (!context->spatial_normalize_cache) {
        context->spatial_normalize_cache = new uint16_t[2 * S + 2];
        for (int x = 0; x < 2 * S + 2; x++) {
            context->spatial_normalize_cache[x] = (uint16_t)(((uint32_t)x * compactness << quantize_level) / S / 2 * 3);
        }
    }
    const uint16_t* spatial_normalize_cache = context->spatial_normalize_cache;

    // Sorting clusters by morton order seems to help for distributing clusters evenly for multiple cores
    std::vector<const Cluster *> cluster_sorted_ptrs;
    for (int k = 0; k < K; k++) { cluster_sorted_ptrs.push_back(&clusters[k]); }

    std::stable_sort(cluster_sorted_ptrs.begin(), cluster_sorted_ptrs.end(), sort_cmp(S));
    // auto t1 = Clock::now();

    // OPTIMIZATION 1: floating point arithmatics is quantized down to int16_t
    // OPTIMIZATION 2: L1 norm instead of L2
    // OPTIMIZATION 5: assignment value is saved combined with distance and cluster number ([distance value (16 bit)] + [cluster number (16 bit)])
    // OPTIMIZATION 6: Make computations of L1 distance SIMD-friendly

    #pragma omp parallel for schedule(static)
    for (int cluster_sorted_idx = 0; cluster_sorted_idx < K; cluster_sorted_idx++) {
        const Cluster *cluster = cluster_sorted_ptrs[cluster_sorted_idx];
        int16_t cluster_y = cluster->y;
        int16_t cluster_x = cluster->x;
        const int16_t y_lo = my_max<int16_t>(0, cluster_y - S), y_hi = my_min<int16_t>(H, cluster_y + S + 1);
        const int16_t x_lo = my_max<int16_t>(0, cluster_x - S), x_hi = my_min<int16_t>(W, cluster_x + S + 1);

        uint16_t row_first_manhattan = (cluster_y - y_lo) + (cluster_x - x_lo);
        for (int16_t i = y_lo; i < cluster_y; i++) {
            uint16_t current_manhattan = row_first_manhattan--;
            #pragma GCC unroll(2)
            for (int16_t j = x_lo; j < cluster_x; j++) {
                int32_t base_index = W * i + j;
                uint16_t spatial_dist = spatial_normalize_cache[current_manhattan--];
                uint32_t assignment_val = get_assignment_value(cluster, image, base_index, spatial_dist, quantize_level);
                if (assignment[base_index] > assignment_val)
                    assignment[base_index] = assignment_val;
            }

            #pragma GCC unroll(2)
            for (int16_t j = cluster_x; j < x_hi; j++) {
                int32_t base_index = W * i + j;
                uint16_t spatial_dist = spatial_normalize_cache[current_manhattan++];
                uint32_t assignment_val = get_assignment_value(cluster, image, base_index, spatial_dist, quantize_level);
                if (assignment[base_index] > assignment_val)
                    assignment[base_index] = assignment_val;
            }
        }

        for (int16_t i = cluster_y; i < y_hi; i++) {
            uint16_t current_manhattan = row_first_manhattan++;
            #pragma GCC unroll(2)
            for (int16_t j = x_lo; j < cluster_x; j++) {
                int32_t base_index = W * i + j;
                uint16_t spatial_dist = spatial_normalize_cache[current_manhattan--];
                uint32_t assignment_val = get_assignment_value(cluster, image, base_index, spatial_dist, quantize_level);
                if (assignment[base_index] > assignment_val)
                    assignment[base_index] = assignment_val;
            }

            #pragma GCC unroll(2)
            for (int16_t j = cluster_x; j < x_hi; j++) {
                int32_t base_index = W * i + j;
                uint16_t spatial_dist = spatial_normalize_cache[current_manhattan++];
                uint32_t assignment_val = get_assignment_value(cluster, image, base_index, spatial_dist, quantize_level);
                if (assignment[base_index] > assignment_val)
                    assignment[base_index] = assignment_val;
            }
        }

    }
    // auto t2 = Clock::now();

    // Clean up: Drop distance part in assignment and let only cluster numbers remain
    #if _OPENMP >= 200805
    #pragma omp parallel for collapse(2)
    #else
    #pragma omp parallel for
    #endif
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            assignment[i * W + j] &= 0x0000FFFF; // drop the leading 2 bytes
        }
    }

    // std::cerr << "Tightloop: " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";

}

static void slic_assign(Context *context) {
    if (!strcmp(context->algorithm, "cluster_oriented")) {
        slic_assign_cluster_oriented(context);
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
            uint32_t max_num_members = 0;
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


static void slic_update_clusters(Context *context) {
    auto H = context->H;
    auto W = context->W;
    auto K = context->K;
    auto image = context->image;
    auto clusters = context->clusters;
    auto assignment = context->assignment;

    int *num_cluster_members = new int[K];
    int *cluster_acc_vec = new int[K * 5]; // sum of [y, x, r, g, b] in cluster

    std::fill_n(num_cluster_members, K, 0);
    std::fill_n(cluster_acc_vec, K * 5, 0);

    #pragma omp parallel
    {
        int *local_acc_vec = new int [K * 5]; // sum of [y, x, r, g, b] in cluster
        int *local_num_cluster_members = new int[K];
        std::fill_n(local_num_cluster_members, K, 0);
        std::fill_n(local_acc_vec, K * 5, 0);
        #if _OPENMP >= 200805
        #pragma omp for collapse(2)
        #else
        #pragma omp for
        #endif
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                int base_index = W * i + j;
                int img_base_index = 3 * base_index;

                cluster_no_t cluster_no = (cluster_no_t)(assignment[base_index]);
                if (cluster_no == 0xFFFF) continue;
                local_num_cluster_members[cluster_no]++;
                local_acc_vec[5 * cluster_no + 0] += i;
                local_acc_vec[5 * cluster_no + 1] += j;
                local_acc_vec[5 * cluster_no + 2] += image[img_base_index];
                local_acc_vec[5 * cluster_no + 3] += image[img_base_index + 1];
                local_acc_vec[5 * cluster_no + 4] += image[img_base_index + 2];
            }
        }

        #pragma omp critical
        {
            for (int k = 0; k < K; k++) {
                for (int dim = 0; dim < 5; dim++) {
                    cluster_acc_vec[5 * k + dim] += local_acc_vec[5 * k + dim];
                }
                num_cluster_members[k] += local_num_cluster_members[k];
            }
        }
        delete [] local_acc_vec;
        delete [] local_num_cluster_members;
    }


    for (int k = 0; k < K; k++) {
        int num_current_members = num_cluster_members[k];
        Cluster *cluster = &clusters[k];
        cluster->num_members = num_current_members;

        if (num_current_members == 0) continue;

        // Technically speaking, as for L1 norm, you need median instead of mean for correct maximization.
        // But, I intentionally used mean here for the sake of performance.
        cluster->y = round_int(cluster_acc_vec[5 * k + 0], num_current_members);
        cluster->x = round_int(cluster_acc_vec[5 * k + 1], num_current_members);
        cluster->r = round_int(cluster_acc_vec[5 * k + 2], num_current_members);
        cluster->g = round_int(cluster_acc_vec[5 * k + 3], num_current_members);
        cluster->b = round_int(cluster_acc_vec[5 * k + 4], num_current_members);
    }
    delete [] num_cluster_members;
    delete [] cluster_acc_vec;
}


extern "C" {
    void fast_slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters) {

        int *gradients = new int[H * W];

        int num_sep = my_max(1, (int)sqrt((double)K));

        int h = H / num_sep;
        int w = W / num_sep;

        // compute gradients
        std::fill_n(gradients, H * W, 1 << 21);
        for (int i = 1; i < H; i += h) {
            for (int j = 1; j < W; j += w) {
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
        for (int i = 0; i < H; i += h) {
            for (int j = 0; j < W; j += w) {
                if (acc_k >= K) break;

                int eh = my_min<int>(i + h, H - 1), ew = my_min<int>(j + w, W - 1);
                int center_y = i + h / 2, center_x = j + w / 2;
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

    void fast_slic_iterate(int H, int W, int K, uint8_t compactness, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint32_t* assignment) {

        Context context;
        init_context(&context);
        context.image = image;
        context.algorithm = "cluster_oriented";
        context.H = H;
        context.W = W;
        context.K = K;
        context.S = (int16_t)sqrt(H * W / K);
        context.compactness = compactness;
        context.quantize_level = quantize_level;
        context.clusters = clusters;
        context.assignment = assignment;

        for (int i = 0; i < max_iter; i++) {
            // auto t1 = Clock::now();
            slic_assign(&context);
            // auto t2 = Clock::now();
            slic_update_clusters(&context);
            // auto t3 = Clock::now();
            // std::cerr << "assignment " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
            // std::cerr << "update "<< std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() << "us \n";
        }

        // auto t1 = Clock::now();
        slic_enforce_connectivity(H, W, K, clusters, assignment);
        // auto t2 = Clock::now();

        // std::cerr << "enforce connectivity "<< std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";


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
    fast_slic_initialize_clusters(H, W, K, image.get(), clusters);
    fast_slic_iterate(H, W, K, compactness, quantize_level, max_iter, image.get(), clusters, assignment.get());

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
