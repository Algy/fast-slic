#include "context.h"
#include "cca.h"
#include "cielab.h"
#include "timer.h"
#include "parallel.h"
#include "tile.h"

#include <limits>
#include <type_traits>

#include "immintrin.h"

static inline __m256i _mm256_abd_epu8(__m256i a, __m256i b) {
    return _mm256_or_si256(_mm256_subs_epu8(a,b), _mm256_subs_epu8(b,a));
}


namespace fslic {
    template<typename DistType>
    BaseContext<DistType>::BaseContext(int H, int W, int K, const uint8_t* image, Cluster *clusters)
            : H(H), W(W), K(K), image(H * W * 3), clusters(clusters), S(sqrt(H * W / K)),
              orig_image(image),
              tile_set(H, W, S),
              dist_patch(S),
              preemptive_grid(H, W, K, S),
              recorder(H, W, K) {

    };
    template<typename DistType>
    BaseContext<DistType>::~BaseContext() {
    }

    template<typename DistType>
    void BaseContext<DistType>::enforce_connectivity(uint16_t *assignment) {
        int thres = (int)round((double)(S * S) * (double)min_size_factor);
        if (K <= 0 || H <= 0 || W <= 0) return;
        cca::ConnectivityEnforcer ce(assignment, H, W, K, thres);
        ce.execute(assignment);
    }

    template<typename DistType>
    void BaseContext<DistType>::initialize_clusters() {
        if (H <= 0 || W <= 0 || K <= 0) return;
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
                clusters[acc_k].is_active = 1;
                clusters[acc_k].is_updatable = 1;

                acc_k++;
            }
        }

        while (acc_k < K) {
            clusters[acc_k].is_active = 1;
            clusters[acc_k].is_updatable = 1;
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
    }

    template<typename DistType>
    void BaseContext<DistType>::initialize_state() {
    }

    template<typename DistType>
    bool BaseContext<DistType>::parallelism_supported() {
        return fsparallel::parallelism_supported();
    }

    template<typename DistType>
    void BaseContext<DistType>::iterate(uint16_t *assignment, int max_iter) {
        {
            fsparallel::Scope parallel_scope(num_threads);
            fstimer::Scope s("iterate");
            {
                fstimer::Scope s("cielab_conversion");
                if (convert_to_lab) {
                    #pragma omp parallel for num_threads(fsparallel::nth())
                    for (int index = 0; index < H * W; index++) {
                        fast_cielab_cvt.convert(
                                orig_image[3 * index],
                                orig_image[3 * index + 1],
                                orig_image[3 * index + 2],
                                image[3 * index],
                                image[3 * index + 1],
                                image[3 * index + 2]
                        );
                    }
                    color_shift = get_cielab_shift();
                } else {
                    std::copy(orig_image, orig_image + image.size(), image.begin());
                    color_shift = 0;
                }
                for (int k = 0; k < K; k++) {
                    int y = clusters[k].y, x = clusters[k].x;
                    y = clamp(y, 0, H - 1);
                    x = clamp(x, 0, W - 1);
                    clusters[k].r = image[3 * (y * W + x)];
                    clusters[k].g = image[3 * (y * W + x) + 1];
                    clusters[k].b = image[3 * (y * W + x) + 2];
                }
            }

            {
                fstimer::Scope s("write_to_buffer");
                tile_set.set_clusters(clusters, K);
                tile_set.set_image(&image[0]);
                tile_set.initialize_dists();

                dist_patch.set(compactness);
            }

            subsample_rem = 0;
            subsample_stride = subsample_stride_config;
            {
                fstimer::Scope s("before_iteration");
                before_iteration();
            }
            preemptive_grid.initialize(clusters, preemptive, preemptive_thres, subsample_stride);
            recorder.initialize(debug_mode);
            // recorder.push(-1, this->assignment, this->min_dists, this->clusters);
            for (int i = 0; i < max_iter; i++) {
                {
                    fstimer::Scope s("assign");
                    assign();
                }

                {
                    fstimer::Scope s("update");
                    update();
                }

                {
                    fstimer::Scope s("after_update");
                    after_update();
                }
                // recorder.push(i, this->assignment, this->min_dists, this->clusters);
                subsample_rem = (subsample_rem + 1) % subsample_stride;
            }
            preemptive_grid.finalize(clusters);

            {
                fstimer::Scope s("full_assign");
                full_assign();
            }
            {
                fstimer::Scope s("write_back");
                tile_set.assign_back(assignment);
            }
            {
                fstimer::Scope s("enforce_connectivity");
                enforce_connectivity(assignment);
            }
        }
        last_timing_report = fstimer::get_report();
    }

    template<typename DistType>
    void BaseContext<DistType>::assign() {
        tile_set.reset_dists();

        // safeguard
        for (int k = 0; k < K; k++) {
            clusters[k].x = clamp<float>(clusters[k].x, 0, W - 1);
            clusters[k].y = clamp<float>(clusters[k].y, 0, H - 1);
        }

        #pragma omp parallel for num_threads(fsparallel::nth())
        for (int tile_no = 0; tile_no < tile_set.get_num_tiles(); tile_no++) {
            assign_clusters(tile_no);
        }
    }


    template<typename DistType>
    void BaseContext<DistType>::full_assign() {
        auto old_subsample_stride = subsample_stride;
        auto old_subsample_rem = subsample_rem;

        subsample_stride = 1;
        subsample_rem = 0;
        assign();
        subsample_stride = old_subsample_stride;
        subsample_rem = old_subsample_rem;
    }

    template<typename DistType>
    void BaseContext<DistType>::assign_clusters(int tile_no) {
        const uint8_t *r_plane = tile_set.get_color_plane(tile_no, 0);
        const uint8_t *g_plane = tile_set.get_color_plane(tile_no, 1);
        const uint8_t *b_plane = tile_set.get_color_plane(tile_no, 2);
        DistType *min_dists = tile_set.get_tile_min_dists(tile_no);
        uint8_t *min_neighbor_indices = tile_set.get_tile_min_neighbor_indices(tile_no);

        auto &neighbors = tile_set.get_neighbor_cluster_nos(tile_no);
        int neighbor_size = neighbors.size();
        int tile_memory_size = tile_set.get_tile_memory_size();

        const auto &tile = tile_set.get_tile(tile_no);

        for (int nid = 0; nid < neighbor_size; nid++) {
            uint16_t cluster_no = neighbors[nid];
            const Cluster* cluster = &clusters[cluster_no];

            int dy = (int)cluster->y - tile.sy, dx = (int)cluster->x - tile.sx;
            if (!(dist_patch.in_range(dy) && dist_patch.in_range(dx))) continue;
            DistType* __restrict spatial_dist_y = dist_patch.at_y(dy);
            DistType* __restrict spatial_dist_x = dist_patch.at_x(dx);

            __m256i cluster_r = _mm256_set1_epi8((uint8_t)cluster->r);
            __m256i cluster_g = _mm256_set1_epi8((uint8_t)cluster->g);
            __m256i cluster_b = _mm256_set1_epi8((uint8_t)cluster->b);
            __m256i cluster_index = _mm256_set1_epi8((uint8_t)nid);

            for (int i = 0; i < tile_memory_size; i += 32) {
                __m256i old_neighbor_vec = _mm256_loadu_si256((__m256i *)&min_neighbor_indices[i]);
                __m256i old_min_dists = _mm256_loadu_si256((__m256i *)&min_dists[i]);
                __m256i r_vec = _mm256_loadu_si256((__m256i *)&r_plane[i]);
                __m256i g_vec = _mm256_loadu_si256((__m256i *)&g_plane[i]);
                __m256i b_vec = _mm256_loadu_si256((__m256i *)&b_plane[i]);
                __m256i dist_r = _mm256_abd_epu8(r_vec, cluster_r);
                __m256i dist_g = _mm256_abd_epu8(g_vec, cluster_g);
                __m256i dist_b = _mm256_abd_epu8(b_vec, cluster_b);
                __m256i dist_color = _mm256_adds_epu8(_mm256_adds_epu8(dist_r, dist_g), dist_b);
                __m256i dist_spatial = _mm256_max_epu8(
                    _mm256_loadu_si256((__m256i *)&spatial_dist_x[i]),
                    _mm256_loadu_si256((__m256i *)&spatial_dist_y[i])
                );
                __m256i dist = _mm256_adds_epu8(dist_color, dist_spatial);
                __m256i new_min_dists = _mm256_min_epu8(old_min_dists, dist);
                // 0xFFFF if a[i+15:i] == b[i+15:i], 0x0000 otherwise.
                __m256i mask = _mm256_cmpeq_epi8(old_min_dists, new_min_dists);
                // if mask[i+7:i] == 0xFF, choose b[i+7:i], otherwise choose a[i+7:i]
                __m256i new_neighbor_vec = _mm256_blendv_epi8(cluster_index, old_neighbor_vec, mask);

                _mm256_storeu_si256((__m256i *)&min_dists[i], new_min_dists);
                _mm256_storeu_si256((__m256i *)&min_neighbor_indices[i], new_neighbor_vec);
            }
        }
    }


    template<typename DistType>
    void BaseContext<DistType>::update() {
        std::vector<int> acc_pool_(fsparallel::nth() * K * 6, 0);
        std::vector<int> cluster_acc_vec_(K * 6, 0);
        int *__restrict acc_pool = &acc_pool_[0];
        int *__restrict cluster_acc_vec = &cluster_acc_vec_[0];

        int T = tile_set.get_num_tiles();

        const int num_threads = fsparallel::nth();


        #pragma omp parallel num_threads(num_threads)
        {
            int *__restrict local_acc_vec = &acc_pool[fsparallel::thindex() * K * 6];
            #pragma omp for
            for (int tile_no = 0; tile_no < T; tile_no++) {
                auto &neighbors = tile_set.get_neighbor_cluster_nos(tile_no);
                int neighbor_size = neighbors.size();
                if (neighbor_size <= 0) continue;
                const Tile &tile = tile_set.get_tile(tile_no);
                int tile_memory_width = tile_set.get_tile_memory_width();
                int tile_memory_size = tile_set.get_tile_memory_size();

                const uint8_t* min_neighbor_indices = tile_set.get_tile_min_neighbor_indices(tile_no);
                const uint8_t *r_plane = tile_set.get_color_plane(tile_no, 0);
                const uint8_t *g_plane = tile_set.get_color_plane(tile_no, 1);
                const uint8_t *b_plane = tile_set.get_color_plane(tile_no, 2);

                for (int y = tile.sy, tile_i_st = 0; y < tile.ey; y++, tile_i_st += tile_memory_width) {
                    for (int x = tile.sx, tile_i = tile_i_st; x < tile.ex; x++, tile_i++) {
                        int neighbor_index = min_neighbor_indices[tile_i];
                        if (neighbor_index == 0xFF) continue;
                        int cluster_no = neighbors[neighbor_index];
                        local_acc_vec[6 * cluster_no + 0]++;
                        local_acc_vec[6 * cluster_no + 1] += y;
                        local_acc_vec[6 * cluster_no + 2] += x;
                        local_acc_vec[6 * cluster_no + 3] += r_plane[tile_i];
                        local_acc_vec[6 * cluster_no + 4] += g_plane[tile_i];
                        local_acc_vec[6 * cluster_no + 5] += b_plane[tile_i];
                    }
                }
            }

            #pragma omp for
            for (int i = 0; i < 6 * K; i++) {
                for (int n = 0; n < num_threads; n++) {
                    cluster_acc_vec[i] += acc_pool[n * (6 * K) + i];
                }
            }

            #pragma omp for
            for (int k = 0; k < K; k++) {
                Cluster *cluster = &clusters[k];
                if (!cluster->is_updatable) continue;
                int num_current_members = cluster_acc_vec[6 * k];
                cluster->num_members = num_current_members;
                if (num_current_members == 0) continue;
                cluster->y = round_int(cluster_acc_vec[6 * k + 1], num_current_members);
                cluster->x = round_int(cluster_acc_vec[6 * k + 2], num_current_members);
                cluster->r = round_int(cluster_acc_vec[6 * k + 3], num_current_members);
                cluster->g = round_int(cluster_acc_vec[6 * k + 4], num_current_members);
                cluster->b = round_int(cluster_acc_vec[6 * k + 5], num_current_members);
            }
        }

        {
            fstimer::Scope s("set_new_clusters");
            preemptive_grid.set_new_clusters(clusters);
        }
    }

    template<typename DistType>
    bool BaseContext<DistType>::centroid_quantization_enabled() {
        return true;
    }

    template class BaseContext<float>;
    template class BaseContext<double>;
    template class BaseContext<uint8_t>;
    template class BaseContext<uint16_t>;
    template class BaseContext<uint32_t>;
};
