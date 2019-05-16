#include <cassert>

#include "fast-slic-neon.h"
#include "fast-slic-common-impl.hpp"

#ifdef USE_NEON
#include <arm_neon.h>

class Context : public BaseContext {
public:
    uint8_t* __restrict__ aligned_quad_image_base = nullptr;
    uint8_t* __restrict__ aligned_quad_image = nullptr; // copied image
    uint16_t quad_image_memory_width;
    uint16_t* __restrict__ aligned_assignment_base = nullptr;
    uint16_t* __restrict__ aligned_assignment = nullptr;
    uint16_t* __restrict__ aligned_min_dists_base = nullptr;
    uint16_t* __restrict__ aligned_min_dists = nullptr;
    int assignment_memory_width; // memory width of aligned_assignment
    int min_dist_memory_width; // memory width of aligned_min_dists;
public:
    virtual ~Context() {
        if (aligned_quad_image_base) {
            simd_helper::free_aligned_array(aligned_quad_image_base);
        }
        if (aligned_assignment_base) {
            simd_helper::free_aligned_array(aligned_assignment_base);
        }
        if (aligned_min_dists_base) {
            simd_helper::free_aligned_array(aligned_min_dists_base);
        }
    }
};

inline void get_assignment_value_vec(
        const Cluster* cluster, const uint16_t* __restrict__ spatial_dist_patch,
        int patch_memory_width,
        int i, int j, int patch_virtual_width,
        const uint8_t* img_quad_row, const uint16_t* spatial_dist_patch_row,
        const uint16_t* min_dist_row, const uint16_t* assignment_row,
        uint16x8_t cluster_number_vec, uint8x16_t cluster_color_vec,
        uint16x8_t& new_min_dist, uint16x8_t& new_assignment
        ) {
    uint16x8_t spatial_dist_vec = vld1q_u16(spatial_dist_patch_row);
#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    {
        for (int delta = 0; delta < 8; delta++) {
            assert(spatial_dist_vec[delta] == spatial_dist_patch[patch_memory_width * i + (j + delta)]);
        }
    }
#endif

    uint8x16_t image_segment = vld1q_u8(img_quad_row);
    uint8x16_t image_segment_2 = vld1q_u8(img_quad_row + 16);

#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    {
        for (int v = 0; v < 4 * my_min(8, patch_virtual_width - j); v++) {
            if (v < 16) {
                if (image_segment[v] != img_quad_row[v]) {
                    abort();
                }
            } else {
                if (image_segment_2[v - 16] != img_quad_row[v]) {
                    abort();
                }
            }
        }
    }
#endif
    uint8x16_t abs_segment = vabdq_u8(image_segment, cluster_color_vec);
    uint8x16_t abs_segment_2 = vabdq_u8(image_segment_2, cluster_color_vec);

    uint32x4_t sad_segment = vpaddlq_u16(vpaddlq_u8(abs_segment));
    uint32x4_t sad_segment_2 = vpaddlq_u16(vpaddlq_u8(abs_segment_2));

    uint16x8_t color_dist_vec = vcombine_u16(vmovn_u32(sad_segment), vmovn_u32(sad_segment_2));

#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    {
        for (int v = 0; v < my_min(8, patch_virtual_width - j); v++) {
            int dr = fast_abs<int>((int)img_quad_row[4 * v + 0] - (int)cluster->r);
            int dg = fast_abs<int>((int)img_quad_row[4 * v + 1] - (int)cluster->g);
            int db= fast_abs<int>((int)img_quad_row[4 * v + 2] - (int)cluster->b);
            int dist = (dr + dg + db) ;
            assert((int)color_dist_vec[v] == dist);
        }
    }
#endif

    uint16x8_t dist_vec = vaddq_u16(color_dist_vec, spatial_dist_vec);
#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    {
        for (int v = 0; v < my_min(8, patch_virtual_width - j); v++) {
            assert(
                    (int)dist_vec[v] ==
                    ((int)spatial_dist_patch[patch_memory_width * i + (j + v)] +
                     ((fast_abs<int>(img_quad_row[4 * v + 0]  - cluster->r) +
                       fast_abs<int>(img_quad_row[4 * v + 1] - cluster->g) +
                       fast_abs<int>(img_quad_row[4 * v + 2] - cluster->b)))
                    )
                  );
        }
    }
#endif

    uint16x8_t old_assignment = vld1q_u16(assignment_row);
    uint16x8_t old_min_dist = vld1q_u16(min_dist_row);
    new_min_dist = vminq_u16(old_min_dist, dist_vec);

    // 0xFFFF if a[i+15:i] == b[i+15:i], 0x0000 otherwise.
    uint16x8_t mask = vceqq_u16(old_min_dist, new_min_dist);
    // if mask[i+15:i] is not zero, choose a[i+15:i], otherwise choose b[i+15:i]
    new_assignment = vbslq_u16(mask, old_assignment, cluster_number_vec);

#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    {
        for (int delta = 0; delta < 8; delta++) {
            assert(cluster_number_vec[delta] == cluster->number);
        }
        for (int delta = 0; delta < 8; delta++) {
            if (old_min_dist[delta] > dist_vec[delta]) {
                assert(new_assignment[delta] == cluster->number);
                assert(new_min_dist[delta] == dist_vec[delta]);
            } else {
                assert(new_assignment[delta] == old_assignment[delta]);
                assert(new_min_dist[delta] == old_min_dist[delta]);
            }
        }
    }
#endif
}

static void slic_assign_cluster_oriented(Context *context) {
    auto H = context->H;
    auto W = context->W;
    auto K = context->K;
    auto clusters = context->clusters;
    auto assignment_memory_width = context->assignment_memory_width;
    auto min_dist_memory_width = context->min_dist_memory_width;
    auto quantize_level = context->quantize_level;
    const int16_t S = context->S;

    const uint8_t* __restrict__ aligned_quad_image = context->aligned_quad_image;
    const uint16_t* __restrict__ spatial_dist_patch = (const uint16_t* __restrict__)HINT_ALIGNED(context->spatial_dist_patch);
    uint16_t* __restrict__ aligned_assignment = context->aligned_assignment;
    uint16_t* __restrict__ aligned_min_dists = context->aligned_min_dists;

    auto quad_image_memory_width = context->quad_image_memory_width;

    // might help to initialize array
#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
    assert((long long)spatial_dist_patch % 32 == 0);
#endif

    const uint16_t patch_height = 2 * S + 1, patch_virtual_width = 2 * S + 1;
    const uint16_t patch_memory_width = simd_helper::align_to_next(patch_virtual_width);

    const uint16x8_t constant = { 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF };
    #pragma omp parallel for
    for (int i = 0; i < H; i++) {
        #pragma unroll(4)
        #pragma GCC unroll(4)
        for (int j = 0; j < W; j += 8) {
            vst1q_u16(&aligned_min_dists[min_dist_memory_width * i + j], constant);
        }
    }

    // Sorting clusters by morton order seems to help for distributing clusters evenly for multiple cores
#   ifdef FAST_SLIC_TIMER
    auto t0 = Clock::now();
#   endif

    std::vector<ZOrderTuple> cluster_sorted_tuples;
    {
        cluster_sorted_tuples.reserve(K);
        for (int k = 0; k < K; k++) {
            const Cluster* cluster = &clusters[k];
            uint32_t score = get_sort_value(cluster->y, cluster->x, S);
            cluster_sorted_tuples.push_back(ZOrderTuple(score, cluster));
        }
        std::sort(cluster_sorted_tuples.begin(), cluster_sorted_tuples.end());
    }

#   ifdef FAST_SLIC_TIMER
    auto t1 = Clock::now();
#   endif
 
    #pragma omp parallel for schedule(static)
    for (int cluster_sorted_idx = 0; cluster_sorted_idx < K; cluster_sorted_idx++) {
        const Cluster *cluster = cluster_sorted_tuples[cluster_sorted_idx].cluster;
        uint16_t cluster_number = cluster->number;
        const uint16_t patch_virtual_width_multiple8 = patch_virtual_width & 0xFFF8;

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

        for (int16_t i = context->fit_to_stride(y_lo) - y_lo; i < patch_height; i += context->subsample_stride) {
            const uint16_t* spatial_dist_patch_base_row = spatial_dist_patch + patch_memory_width * i;
#ifdef FAST_SLIC_SIMD_INVARIANCE_CHECK
            assert((long long)spatial_dist_patch_base_row % 32 == 0);
#endif
            // not aligned
            const uint8_t *img_quad_base_row = aligned_quad_image + quad_image_memory_width * (y_lo + i) + 4 * x_lo;
            uint16_t* assignment_base_row = aligned_assignment + (i + y_lo) * assignment_memory_width + x_lo;
            uint16_t* min_dist_base_row = aligned_min_dists + (i + y_lo) * min_dist_memory_width + x_lo;

#define ASSIGNMENT_VALUE_GETTER_BODY \
    uint16x8_t new_min_dist, new_assignment; \
    uint16_t* min_dist_row = min_dist_base_row + j; /* unaligned */ \
    uint16_t* assignment_row = assignment_base_row + j;  /* unaligned */ \
    const uint8_t* img_quad_row = img_quad_base_row + 4 * j; /*Image rows are not aligned due to x_lo*/ \
    const uint16_t* spatial_dist_patch_row = (uint16_t *)HINT_ALIGNED_AS(spatial_dist_patch_base_row + j, 16); /* Spatial distance patch is aligned */ \
    get_assignment_value_vec( \
        cluster, \
        spatial_dist_patch, \
        patch_memory_width, \
        i, j, patch_virtual_width, \
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
            for (int j = 0; j < patch_virtual_width_multiple8; j += 8) {
                ASSIGNMENT_VALUE_GETTER_BODY
                vst1q_u16(min_dist_row, new_min_dist);
                vst1q_u16(assignment_row, new_assignment);
            }

            if (0 < patch_virtual_width - patch_virtual_width_multiple8) {
                uint16_t new_min_dists[8], new_assignments[8];
                int j = patch_virtual_width_multiple8;
                ASSIGNMENT_VALUE_GETTER_BODY
                vst1q_u16(new_min_dists, new_min_dist);
                vst1q_u16(new_assignments, new_assignment);
                for (int delta = 0; delta < patch_virtual_width - patch_virtual_width_multiple8; delta++) {
                    min_dist_row[delta] = new_min_dists[delta];
                    assignment_row[delta] = new_assignments[delta];
                }
            }
        }
    }
#   ifdef FAST_SLIC_TIMER
    auto t2 = Clock::now();
    std::cerr << "Sort: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << "us \n";
    std::cerr << "Tightloop: " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
#   endif
}

static void slic_assign(Context *context) {
    if (!strcmp(context->algorithm, "cluster_oriented")) {
        slic_assign_cluster_oriented(context);
    }
}

static void slic_update_clusters(Context *context) {
    auto H = context->H;
    auto W = context->W;
    auto K = context->K;
    auto aligned_quad_image = context->aligned_quad_image;
    auto clusters = context->clusters;
    auto aligned_assignment = context->aligned_assignment;
    auto quad_image_memory_width = context->quad_image_memory_width;
    auto assignment_memory_width = context->assignment_memory_width;

    int *num_cluster_members = new int[K];
    int *cluster_acc_vec = new int[K * 5]; // sum of [y, x, r, g, b] in cluster

    std::fill_n(num_cluster_members, K, 0);
    std::fill_n((int *)cluster_acc_vec, K * 5, 0);

    #pragma omp parallel
    {
        uint32_t *local_acc_vec = new uint32_t[K * 5]; // sum of [y, x, r, g, b] in cluster
        int *local_num_cluster_members = new int[K];
        std::fill_n(local_num_cluster_members, K, 0);
        std::fill_n(local_acc_vec, K * 5, 0);

        #pragma omp for
        for (int i = context->fit_to_stride(0); i < H; i += context->subsample_stride) {
            for (int j = 0; j < W; j++) {
                int img_base_index = quad_image_memory_width * i + 4 * j;
                int assignment_index = assignment_memory_width * i + j;

                uint16_t cluster_no = aligned_assignment[assignment_index];
                if (cluster_no == 0xFFFF) continue;
                local_num_cluster_members[cluster_no]++;
                local_acc_vec[5 * cluster_no + 0] += i;
                local_acc_vec[5 * cluster_no + 1] += j;
                local_acc_vec[5 * cluster_no + 2] += aligned_quad_image[img_base_index];
                local_acc_vec[5 * cluster_no + 3] += aligned_quad_image[img_base_index + 1];
                local_acc_vec[5 * cluster_no + 4] += aligned_quad_image[img_base_index + 2];
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

        delete [] local_num_cluster_members;
        delete [] local_acc_vec;
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
    void fast_slic_initialize_clusters_neon(int H, int W, int K, const uint8_t* image, Cluster *clusters) {
#       ifdef FAST_SLIC_TIMER
        auto t1 = Clock::now();
#       endif
        do_fast_slic_initialize_clusters(H, W, K, image, clusters);
#       ifdef FAST_SLIC_TIMER
        auto t2 = Clock::now();
        std::cerr << "Cluster Initialization: " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
#       endif
    }

    void fast_slic_iterate_neon(int H, int W, int K, float compactness, float min_size_factor, uint8_t quantize_level, int max_iter, const uint8_t *__restrict__ image, Cluster *__restrict__ clusters, uint16_t* __restrict__ assignment) {
        int S = sqrt(H * W / K);

        Context context;
        context.image = image;
        context.algorithm = "cluster_oriented";
        context.H = H;
        context.W = W;
        context.K = K;
        context.S = (int16_t)S;
        context.assignment = assignment;
        context.compactness = compactness;
        context.min_size_factor = min_size_factor;
        context.quantize_level = quantize_level;
        context.clusters = clusters;

        // Pad image and assignment
        uint32_t quad_image_memory_width;
        context.quad_image_memory_width = quad_image_memory_width = simd_helper::align_to_next((W + 2 * S) * 4);
        uint8_t* aligned_quad_image_base = simd_helper::alloc_aligned_array<uint8_t>((H + 2 * S) * quad_image_memory_width);


        context.aligned_quad_image_base = aligned_quad_image_base;
        context.aligned_quad_image = &aligned_quad_image_base[quad_image_memory_width * S + S * 4];

        context.assignment_memory_width = simd_helper::align_to_next(W + 2 * S);
        context.aligned_assignment_base = simd_helper::alloc_aligned_array<uint16_t>((H + 2 * S) * context.assignment_memory_width);
        context.aligned_assignment = &context.aligned_assignment_base[S * context.assignment_memory_width + S];

        context.min_dist_memory_width = context.assignment_memory_width;
        context.aligned_min_dists_base = simd_helper::alloc_aligned_array<uint16_t>((H + 2 * S) * context.min_dist_memory_width);
        context.aligned_min_dists = &context.aligned_min_dists_base[S * context.min_dist_memory_width + S];

        context.prepare_spatial();
        {
#           ifdef FAST_SLIC_TIMER
            auto t1 = Clock::now();
#           endif

            #pragma omp parallel for
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    for (int k = 0; k < 3; k++) {
                        aligned_quad_image_base[(i + S) * quad_image_memory_width + 4 * (j + S) + k] = image[i * W * 3 + 3 * j + k];
                    }
                }
            }

            const uint16x8_t constant = { 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF };
            #pragma omp parallel for
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j += 8) {
                    vst1q_u16(&context.aligned_assignment[context.assignment_memory_width * i + j], constant);
                }
            }

#           ifdef FAST_SLIC_TIMER
            auto t2 = Clock::now();
            std::cerr << "Assignment Initialization " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
#           endif
        }


        for (int i = 0; i < max_iter; i++) {
            if (i == max_iter - 1) {
                context.subsample_stride = 1;
                context.subsample_rem = 0;
            } else {
                context.subsample_rem++;
                context.subsample_rem %= context.subsample_stride;
            }

#           ifdef FAST_SLIC_TIMER
            auto t1 = Clock::now();
#           endif
            slic_assign(&context);
#           ifdef FAST_SLIC_TIMER
            auto t2 = Clock::now();
#           endif
            slic_update_clusters(&context);
#           ifdef FAST_SLIC_TIMER
            auto t3 = Clock::now();
            std::cerr << "assignment " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
            std::cerr << "update "<< std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() << "us \n";
#           endif
        }



        {
#           ifdef FAST_SLIC_TIMER
            auto t1 = Clock::now();
#           endif

            #pragma omp parallel for
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    assignment[W * i + j] = context.aligned_assignment[context.assignment_memory_width * i + j];
                }
            }
#           ifdef FAST_SLIC_TIMER
            auto t2 = Clock::now();
            std::cerr << "Write back assignment"<< std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
#           endif
        }
#       ifdef FAST_SLIC_TIMER
        auto t1 = Clock::now();
#       endif
        slic_enforce_connectivity(&context);
#       ifdef FAST_SLIC_TIMER
        auto t2 = Clock::now();
        std::cerr << "enforce connectivity "<< std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
#       endif
    }
    int fast_slic_supports_neon() { return 1; }
}

#else // else of #ifdef USE_NEON

extern "C" {
    void fast_slic_initialize_clusters_neon(int H, int W, int K, const uint8_t* image, Cluster *clusters) {}
    void fast_slic_iterate_neon(int H, int W, int K, float compactness, float min_size_factor, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint16_t* assignment) {}
int fast_slic_supports_neon() { return 0; }
}

#endif // of #ifdef USE_NEON


#ifdef PROTOTYPE_MAIN_DEMO

#ifndef USE_NEON
#error "Compile it with flag USE_NEON"
#endif 

#include <cstdlib>
#include <ctime>
#include <string>
#include <chrono>
#include <fstream>
#include <memory>
typedef std::chrono::high_resolution_clock Clock;
int main(int argc, char** argv) {
    int K = 100;
    int compactness = 5;
    int max_iter = 2;
    int quantize_level = 6;
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
    srand(time(nullptr));

    Cluster clusters[K];
    std::unique_ptr<uint8_t[]> image { new uint8_t[H * W * 3] };
    std::unique_ptr<uint16_t[]> assignment { new uint16_t[H * W] };

    /*
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
    */
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int r, g, b;
            r = (int)(rand() * 255);
            g = (int)(rand() * 255);
            b = (int)(rand() * 255);
            image.get()[3 * W * i + 3 * j] = r;
            image.get()[3 * W * i + 3 * j + 1] = g;
            image.get()[3 * W * i + 3 * j + 2] = b;
        }
    }

    auto t1 = Clock::now();
    fast_slic_initialize_clusters_neon(H, W, K, image.get(), clusters);
    fast_slic_iterate_neon(H, W, K, compactness, 0.1, quantize_level, max_iter, image.get(), clusters, assignment.get());

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
