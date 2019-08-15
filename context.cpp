#include "context.h"
#include "simd-helper.hpp"
#include "cca.h"
#include "cielab.h"

#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef _OPENMP
#define omp_get_num_threads() 1
#endif

// #define FAST_SLIC_TIMER

namespace fslic {
    template<typename DistType>
    BaseContext<DistType>::~BaseContext() {
        if (spatial_dist_patch) {
            simd_helper::free_aligned_array(spatial_dist_patch);
        }
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

    template<typename DistType>
    void BaseContext<DistType>::enforce_connectivity(uint16_t *assignment) {
        int thres = (int)round((double)(S * S) * (double)min_size_factor);
        if (K <= 0 || H <= 0 || W <= 0) return;
        cca::ConnectivityEnforcer ce(assignment, H, W, K, thres);
        ce.execute(assignment);
    }

    template <typename DistType>
    void BaseContext<DistType>::prepare_spatial() {
        patch_height = patch_virtual_width = 2 * S + 1;
        patch_memory_width = simd_helper::align_to_next(patch_virtual_width);

        if (spatial_dist_patch) simd_helper::free_aligned_array(spatial_dist_patch);
        spatial_dist_patch = simd_helper::alloc_aligned_array<DistType>(patch_height * patch_memory_width);
        set_spatial_patch();
    }

    template<typename DistType>
    void BaseContext<DistType>::set_spatial_patch() {
        float coef = 1.0f / ((float)S / compactness);
        int16_t S_2 = 2 * S;
        for (int16_t i = 0; i <= S_2; i++) {
            for (int16_t j = 0; j <= S_2; j++) {
                spatial_dist_patch[i * patch_memory_width + j] = (DistType)(coef * (fast_abs(i - S) + fast_abs(j - S)));
            }
        }
    }

    template<typename DistType>
    void BaseContext<DistType>::initialize_clusters() {
        if (H <= 0 || W <= 0 || K <= 0) return;
    #ifdef FAST_SLIC_TIMER
        auto t1 = Clock::now();
    #endif
        int n_y = (int)sqrt((double)K);

        std::vector<int> n_xs(n_y, K / n_y);

        int remainder = K % n_y;
        int row = 0;
        while (remainder-- > 0) {
            n_xs[row]++;
            row += 2;
            if (row >= n_y) {
                row = 1 % n_y;
            }
        }

        int h = ceil_int(H, n_y);
        int acc_k = 0;
        for (int i = 0; i < H; i += h) {
            int w = ceil_int(W, n_xs[my_min<int>(i / h, n_y - 1)]);
            for (int j = 0; j < W; j += w) {
                if (acc_k >= K) {
                    break;
                }
                int center_y = i + h / 2, center_x = j + w / 2;
                center_y = clamp(center_y, 0, H - 1);
                center_x = clamp(center_x, 0, W - 1);

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

        for (int k = 0; k < K; k++) {
            int base_index = W * clusters[k].y + clusters[k].x;
            int img_base_index = 3 * base_index;
            clusters[k].r = image[img_base_index];
            clusters[k].g = image[img_base_index + 1];
            clusters[k].b = image[img_base_index + 2];
            clusters[k].number = k;
            clusters[k].num_members = 0;
        }
        #ifdef FAST_SLIC_TIMER
        auto t2 = Clock::now();
        std::cerr << "Cluster initialization: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us\n";

        #endif
    }

    template<typename DistType>
    void BaseContext<DistType>::initialize_state() {
#       ifdef FAST_SLIC_TIMER
        auto t0 = Clock::now();
#       endif
        quad_image_memory_width = simd_helper::align_to_next((W + 2 * S) * 4);
        aligned_quad_image_base = simd_helper::alloc_aligned_array<uint8_t>((H + 2 * S) * quad_image_memory_width);
        aligned_quad_image = &aligned_quad_image_base[quad_image_memory_width * S + S * 4];

        assignment_memory_width = simd_helper::align_to_next(W + 2 * S);
        aligned_assignment_base = simd_helper::alloc_aligned_array<uint16_t>((H + 2 * S) * assignment_memory_width);
        aligned_assignment = &aligned_assignment_base[S * assignment_memory_width + S];

        min_dist_memory_width = assignment_memory_width;
        aligned_min_dists_base = simd_helper::alloc_aligned_array<DistType>((H + 2 * S) * min_dist_memory_width);
        aligned_min_dists = &aligned_min_dists_base[S * min_dist_memory_width + S];

        prepare_spatial();
#       ifdef FAST_SLIC_TIMER
        auto t1 = Clock::now();
        std::cerr << "Initialize spatial map: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << "us\n";
#       endif
    }

    template<typename DistType>
    bool BaseContext<DistType>::parallelism_supported() {
#if defined(_OPENMP)
    return true;
#else
    return false;
#endif
    }

    template<typename DistType>
    void BaseContext<DistType>::iterate(uint16_t *assignment, int max_iter) {
        {
#           ifdef FAST_SLIC_TIMER
            auto t0 = Clock::now();
#           endif
            #pragma omp parallel
            {
                #pragma omp for
                for (int i = 0; i < H; i++) {
                    for (int j = 0; j < W; j++) {
                        for (int k = 0; k < 3; k++) {
                            aligned_quad_image_base[(i + S) * quad_image_memory_width + 4 * (j + S) + k] = image[i * W * 3 + 3 * j + k];
                        }
                    }
                }

                #pragma omp for
                for (int i = 0; i < H; i++) {
                    for (int j = 0; j < W; j++) {
                        aligned_assignment[assignment_memory_width * i + j] = 0xFFFF;
                    }
                }
            }

#           ifdef FAST_SLIC_TIMER
            auto t1 = Clock::now();
            std::cerr << "Copy Image&initialize label map: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << "us\n";
#           endif
        }

        if (convert_to_lab) {
            rgb_to_cielab(aligned_quad_image, aligned_quad_image, quad_image_memory_width * H, true);
        }

        // Pad image and assignment
        subsample_rem = 1;
        subsample_stride = subsample_stride_config;

        for (int i = 0; i < max_iter; i++) {
            if (i == max_iter - 1) {
                subsample_stride = 1;
                subsample_rem = 0;
            } else {
                subsample_rem++;
                subsample_rem %= subsample_stride;
            }

#           ifdef FAST_SLIC_TIMER
            auto t1 = Clock::now();
#           endif
            assign();
#           ifdef FAST_SLIC_TIMER
            auto t2 = Clock::now();
#           endif
            update();
#           ifdef FAST_SLIC_TIMER
            auto t3 = Clock::now();
            std::cerr << "assignment " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
            std::cerr << "update "<< std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() << "us \n";
#           endif
        }
        // assign();

        {
#           ifdef FAST_SLIC_TIMER
            auto t1 = Clock::now();
#           endif

            #pragma omp parallel for
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    assignment[W * i + j] = aligned_assignment[assignment_memory_width * i + j];
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
        enforce_connectivity(assignment);
#       ifdef FAST_SLIC_TIMER
        auto t2 = Clock::now();
        std::cerr << "enforce connectivity "<< std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
#       endif
    }

    template<typename DistType>
    void BaseContext<DistType>::assign() {
        #pragma omp parallel for
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                aligned_min_dists[min_dist_memory_width * i + j] = std::numeric_limits<DistType>::max();
            }
        }

        // safeguard
        for (int k = 0; k < K; k++) {
            clusters[k].x = clamp<int16_t>(clusters[k].x, 0, W - 1);
            clusters[k].y = clamp<int16_t>(clusters[k].y, 0, H - 1);
        }

        int T = 2 * S;
        int cell_W = ceil_int(W, T), cell_H = ceil_int(H, T);
        std::vector< std::vector<const Cluster*> > grid(cell_W * cell_H);
        for (int k = 0; k < K; k++) {
            int y = clusters[k].y, x = clusters[k].x;
            grid[cell_W * (y / T) + (x / T)].push_back(&clusters[k]);
        }

        for (int phase = 0; phase < 4; phase++) {
            std::vector<int> grid_indices;
            for (int i = phase / 2; i < cell_H; i += 2) {
                for (int j = phase % 2; j < cell_W; j += 2) {
                    grid_indices.push_back(i * cell_W + j);
                }
            }
            #pragma omp parallel
            {
                std::vector<const Cluster*> target_clusters;
                #pragma omp for
                for (int cell_ix = 0; cell_ix < (int)grid_indices.size(); cell_ix++) {
                    const std::vector<const Cluster*> &clusters_in_cell = grid[grid_indices[cell_ix]];
                    for (int inst = 0; inst < (int)clusters_in_cell.size(); inst++) {
                        target_clusters.push_back(clusters_in_cell[inst]);
                    }
                }
                assign_clusters(&target_clusters[0], (int)target_clusters.size());
            }
        }
    }

    template<typename DistType>
    void BaseContext<DistType>::assign_clusters(const Cluster** target_clusters, int size) {
        DistType* __restrict dist_row = new DistType[2 * S + 1];

        const int16_t S_2 = 2 * S;

        for (int cidx = 0; cidx < size; cidx++) {
            const Cluster* cluster = target_clusters[cidx];
            int16_t cluster_y = cluster->y, cluster_x = cluster->x;
            int16_t cluster_r = cluster->r, cluster_g = cluster->g, cluster_b = cluster->b;
            uint16_t cluster_no = cluster->number;

            for (int16_t i_off = 0, i = cluster_y - S; i_off <= S_2; i_off++, i++) {
                if (!valid_subsample_row(i)) continue;
                const uint8_t __restrict *image_row = &aligned_quad_image[quad_image_memory_width * i + 4 * (cluster_x - S)];
                uint16_t __restrict  *assignment_row = &aligned_assignment[assignment_memory_width * i + (cluster_x - S)];
                DistType __restrict  *min_dist_row = &aligned_min_dists[min_dist_memory_width * i + (cluster_x - S)];
                const DistType __restrict *patch_row = &spatial_dist_patch[patch_memory_width * i_off];

                for (int16_t j_off = 0; j_off <= S_2; j_off++) {
                    dist_row[j_off] = patch_row[j_off];
                }

                for (int16_t j_off = 0; j_off <= S_2; j_off++) {
                    int16_t r = image_row[4 * j_off],
                        g = image_row[4 * j_off + 1],
                        b = image_row[4 * j_off + 2];
                    DistType color_dist = fast_abs(r - cluster_r) + fast_abs(g - cluster_g) + fast_abs(b - cluster_b);
                    dist_row[j_off] += color_dist;
                }

                for (int16_t j_off = 0; j_off <= S_2; j_off++) {
                    if (min_dist_row[j_off] > dist_row[j_off]) {
                        min_dist_row[j_off] = dist_row[j_off];
                        assignment_row[j_off] = cluster_no;
                    }
                }
            }
        }
        delete [] dist_row;
    }


    template<typename DistType>
    void BaseContext<DistType>::update() {
        std::vector<int32_t> num_cluster_members(K, 0);
        std::vector<int32_t> cluster_acc_vec(K * 5, 0); // sum of [y, x, r, g, b] in cluster

        #pragma omp parallel
        {
            std::vector<uint32_t> local_acc_vec(K * 5, 0); // sum of [y, x, r, g, b] in cluster
            std::vector<uint32_t> local_num_cluster_members(K, 0);
            #pragma omp for
            for (int i = fit_to_stride(0); i < H; i += subsample_stride) {
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
                for (int i = 0; i < (int)local_acc_vec.size(); i++) {
                    cluster_acc_vec[i] += local_acc_vec[i];
                }
                for (int k = 0; k < K; k++) {
                    num_cluster_members[k] += local_num_cluster_members[k];
                }
            }
        }


        for (int k = 0; k < K; k++) {
            int32_t num_current_members = num_cluster_members[k];
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
    }

    void ContextRealDistL2::assign_clusters(const Cluster** target_clusters, int size) {
        float* dist_row = new float[2 * S + 1];

        const int16_t S_2 = 2 * S;

        for (int cidx = 0; cidx < size; cidx++) {
            const Cluster* cluster = target_clusters[cidx];
            int16_t cluster_y = cluster->y, cluster_x = cluster->x;
            int16_t cluster_r = cluster->r, cluster_g = cluster->g, cluster_b = cluster->b;
            uint16_t cluster_no = cluster->number;

            for (int16_t i_off = 0, i = cluster_y - S; i_off <= S_2; i_off++, i++) {
                if (!valid_subsample_row(i)) continue;
                const uint8_t *image_row = &aligned_quad_image[quad_image_memory_width * i + 4 * (cluster_x - S)];
                uint16_t *assignment_row = &aligned_assignment[assignment_memory_width * i + (cluster_x - S)];
                float *min_dist_row = &aligned_min_dists[min_dist_memory_width * i + (cluster_x - S)];
                const float *patch_row = &spatial_dist_patch[patch_memory_width * i_off];

                for (int16_t j_off = 0; j_off <= S_2; j_off++) {
                    dist_row[j_off] = patch_row[j_off];
                }

                for (int16_t j_off = 0; j_off <= S_2; j_off++) {
                    float dr = image_row[4 * j_off] - cluster_r,
                        dg = image_row[4 * j_off + 1] - cluster_g,
                        db = image_row[4 * j_off + 2] - cluster_b;
                    float color_dist = dr*dr + dg*dg + db*db;
                    dist_row[j_off] += color_dist;
                }

                for (int16_t j_off = 0; j_off <= S_2; j_off++) {
                    if (min_dist_row[j_off] > dist_row[j_off]) {
                        min_dist_row[j_off] = dist_row[j_off];
                        assignment_row[j_off] = cluster_no;
                    }
                }
            }
        }
        delete [] dist_row;
    }

    void ContextRealDistL2::set_spatial_patch() {
        float coef = 1.0f / ((float)S / compactness);
        int16_t S_2 = 2 * S;
        for (int16_t i = 0; i <= S_2; i++) {
            for (int16_t j = 0; j <= S_2; j++) {
                float di = coef * (i - S), dj = coef * (j - S);
                spatial_dist_patch[i * patch_memory_width + j] = di * di + dj * dj;
            }
        }
    }

    template class BaseContext<float>;
    template class BaseContext<double>;
    template class BaseContext<uint16_t>;
    template class BaseContext<uint32_t>;
};
