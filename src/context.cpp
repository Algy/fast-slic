#include "context.h"
#include "cca.h"
#include "cielab.h"
#include "timer.h"
#include "parallel.h"

#include <limits>

namespace fslic {
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
    void BaseContext<DistType>::set_spatial_patch() {
        float coef = 1.0f / ((float)S / compactness);
        coef *= (1 << color_shift);
        int16_t S_2 = 2 * S;
        if (manhattan_spatial_dist) {
            for (int16_t i = 0; i <= S_2; i++) {
                for (int16_t j = 0; j <= S_2; j++) {
                    spatial_dist_patch.get(i, j) = (DistType)(coef * (fast_abs(i - S) + fast_abs(j - S)));
                }
            }
        } else {
            for (int16_t i = 0; i <= S_2; i++) {
                for (int16_t j = 0; j <= S_2; j++) {
                    spatial_dist_patch.get(i, j) = (DistType)(coef * hypotf(i - S, j - S));
                }
            }
        }
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
                    rgb_to_cielab(image, H, W, quad_image, color_shift);
                } else {
                    #pragma omp parallel num_threads(fsparallel::nth())
                    for (int i = 0; i < H; i++) {
                        for (int j = 0; j < W; j++) {
                            for (int k = 0; k < 3; k++) {
                                quad_image.get(i, 4 * j + k) = image[i * W * 3 + 3 * j + k];
                            }
                        }
                    }
                    color_shift = 0;
                }
                for (int k = 0; k < K; k++) {
                    int y = clusters[k].y, x = clusters[k].x;
                    y = clamp(y, 0, H - 1);
                    x = clamp(x, 0, W - 1);
                    clusters[k].r = quad_image.get(y, 4 * x);
                    clusters[k].g = quad_image.get(y, 4 * x + 1);
                    clusters[k].b = quad_image.get(y, 4 * x + 2);
                }
            }

            {
                fstimer::Scope s("write_to_buffer");
                #pragma omp parallel for num_threads(fsparallel::nth())
                for (int i = 0; i < H; i++) {
                    for (int j = 0; j < W; j++) {
                        this->assignment.get(i, j) = 0xFFFF;
                    }
                }
                set_spatial_patch();
            }

            subsample_rem = 0;
            subsample_stride = subsample_stride_config;
            {
                fstimer::Scope s("before_iteration");
                before_iteration();
            }
            preemptive_grid.initialize(clusters, preemptive, preemptive_thres, subsample_stride);
            recorder.initialize(debug_mode);
            recorder.push(-1, this->assignment, this->min_dists, this->clusters);
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
                recorder.push(i, this->assignment, this->min_dists, this->clusters);
                subsample_rem = (subsample_rem + 1) % subsample_stride;
            }
            preemptive_grid.finalize(clusters);

            {
                fstimer::Scope s("full_assign");
                full_assign();
            }
            {
                fstimer::Scope s("write_back");
                #pragma omp parallel for num_threads(fsparallel::nth())
                for (int i = 0; i < H; i++) {
                    for (int j = 0; j < W; j++) {
                        assignment[W * i + j] = this->assignment.get(i, j);
                    }
                }
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
        #pragma omp parallel for num_threads(fsparallel::nth())
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                min_dists.get(i, j) = std::numeric_limits<DistType>::max();
            }
        }

        // safeguard
        for (int k = 0; k < K; k++) {
            clusters[k].x = clamp<float>(clusters[k].x, 0, W - 1);
            clusters[k].y = clamp<float>(clusters[k].y, 0, H - 1);
        }

        int T = 2 * S + 32;
        int cell_W = ceil_int(W, T), cell_H = ceil_int(H, T);
        std::vector< std::vector<const Cluster*> > grid(cell_W * cell_H);
        for (int k = 0; k < K; k++) {
            if (!clusters[k].is_active) continue;
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
            #pragma omp parallel num_threads(fsparallel::nth())
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
    void BaseContext<DistType>::assign_clusters(const Cluster** target_clusters, int size) {
        DistType* __restrict dist_row = new DistType[2 * S + 1];

        const int16_t S_2 = 2 * S;

        for (int cidx = 0; cidx < size; cidx++) {
            const Cluster* cluster = target_clusters[cidx];
            int16_t cluster_y = cluster->y, cluster_x = cluster->x;
            int16_t cluster_r = cluster->r, cluster_g = cluster->g, cluster_b = cluster->b;
            uint16_t cluster_no = cluster->number;

            for (int i_off = 0, i = cluster_y - S; i_off <= S_2; i_off++, i++) {
                if (!valid_subsample_row(i)) continue;
                const uint8_t* __restrict image_row = quad_image.get_row(i, 4 * (cluster_x - S));
                uint16_t* __restrict  assignment_row = assignment.get_row(i, cluster_x - S);
                DistType* __restrict min_dist_row = min_dists.get_row(i, cluster_x - S);
                const DistType* __restrict patch_row = spatial_dist_patch.get_row(i_off);

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
        preemptive_grid.set_old_clusters(clusters);

        std::vector<int32_t> num_cluster_members(K, 0);
        std::vector<int32_t> cluster_acc_vec(K * 5, 0); // sum of [y, x, r, g, b] in cluster
        std::vector<PreemptiveTile> active_tiles = preemptive_grid.get_active_tiles();

        #pragma omp parallel num_threads(fsparallel::nth())
        {
            std::vector<uint32_t> local_acc_vec(K * 5, 0); // sum of [y, x, r, g, b] in cluster
            std::vector<uint32_t> local_num_cluster_members(K, 0);
            // if a cell is active, it is updatable (but not vice versa).
            if (preemptive_grid.all_active()) {
                #pragma omp for
                for (int i = fit_to_stride(0); i < H; i += subsample_stride) {
                    for (int j = 0; j < W; j++) {
                        uint16_t cluster_no = assignment.get(i, j);
                        if (cluster_no == 0xFFFF) continue;
                        local_num_cluster_members[cluster_no]++;
                        local_acc_vec[5 * cluster_no + 0] += i;
                        local_acc_vec[5 * cluster_no + 1] += j;
                        local_acc_vec[5 * cluster_no + 2] += quad_image.get(i, 4*j);
                        local_acc_vec[5 * cluster_no + 3] += quad_image.get(i, 4*j + 1);
                        local_acc_vec[5 * cluster_no + 4] += quad_image.get(i, 4*j + 2);
                    }
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (int i = fit_to_stride(0); i < H; i += subsample_stride) {
                    for (int j = 0; j < W; j++) {
                        if (!preemptive_grid.get_active_cell(i, j)) continue;
                        uint16_t cluster_no = assignment.get(i, j);
                        if (cluster_no == 0xFFFF) continue;
                        local_num_cluster_members[cluster_no]++;
                        local_acc_vec[5 * cluster_no + 0] += i;
                        local_acc_vec[5 * cluster_no + 1] += j;
                        local_acc_vec[5 * cluster_no + 2] += quad_image.get(i, 4*j);
                        local_acc_vec[5 * cluster_no + 3] += quad_image.get(i, 4*j + 1);
                        local_acc_vec[5 * cluster_no + 4] += quad_image.get(i, 4*j + 2);
                    }
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


        bool centroid_qt = centroid_quantization_enabled();
        for (int k = 0; k < K; k++) {
            Cluster *cluster = &clusters[k];
            if (!cluster->is_updatable) continue;
            int32_t num_current_members = num_cluster_members[k];
            cluster->num_members = num_current_members;

            if (num_current_members == 0) continue;

            // Technically speaking, as for L1 norm, you need median instead of mean for correct maximization.
            // But, I intentionally used mean here for the sake of performance.
            if (centroid_qt) {
                cluster->y = round_int(cluster_acc_vec[5 * k + 0], num_current_members);
                cluster->x = round_int(cluster_acc_vec[5 * k + 1], num_current_members);
                cluster->r = round_int(cluster_acc_vec[5 * k + 2], num_current_members);
                cluster->g = round_int(cluster_acc_vec[5 * k + 3], num_current_members);
                cluster->b = round_int(cluster_acc_vec[5 * k + 4], num_current_members);
            } else {
                cluster->y = (float)cluster_acc_vec[5 * k + 0] / num_current_members;
                cluster->x = (float)cluster_acc_vec[5 * k + 1] / num_current_members;
                cluster->r = (float)cluster_acc_vec[5 * k + 2] / num_current_members;
                cluster->g = (float)cluster_acc_vec[5 * k + 3] / num_current_members;
                cluster->b = (float)cluster_acc_vec[5 * k + 4] / num_current_members;
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
                const uint8_t* __restrict image_row = quad_image.get_row(i, 4 * (cluster_x - S));
                uint16_t* __restrict assignment_row = assignment.get_row(i, cluster_x - S);
                float* __restrict  min_dist_row = min_dists.get_row(i, cluster_x - S);
                const float* __restrict patch_row = spatial_dist_patch.get_row(i_off);

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
        coef *= (1 << color_shift);
        int16_t S_2 = 2 * S;
        for (int16_t i = 0; i <= S_2; i++) {
            for (int16_t j = 0; j <= S_2; j++) {
                float di = coef * (i - S), dj = coef * (j - S);
                spatial_dist_patch.get(i, j) = di * di + dj * dj;
            }
        }
    }

    void ContextRealDistNoQ::before_iteration() {
    }

    bool ContextRealDistNoQ::centroid_quantization_enabled() {
        return false;
    }

    void ContextRealDistNoQ::assign_clusters(const Cluster** target_clusters, int size) {
        if (manhattan_spatial_dist) {
            assign_clusters_proto<true>(target_clusters, size);
        } else {
            assign_clusters_proto<false>(target_clusters, size);
        }
    }

    template<bool use_manhattan>
    void ContextRealDistNoQ::assign_clusters_proto(const Cluster** target_clusters, int size) {
        float coef = 1.0f / ((float)S / compactness);
        coef *= (1 << color_shift);

        for (int cidx = 0; cidx < size; cidx++) {
            const Cluster* cluster = target_clusters[cidx];
            float cluster_y = cluster->y, cluster_x = cluster->x;
            float cluster_r = cluster->r, cluster_g = cluster->g, cluster_b = cluster->b;
            int y_lo = my_max<int>(cluster_y - S, 0), y_hi = my_min<int>(cluster_y + S + 1, H);
            int x_lo = my_max<int>(cluster_x - S, 0), x_hi = my_min<int>(cluster_x + S + 1, W);

            uint16_t cluster_no = cluster->number;
            for (int i = y_lo; i < y_hi; i++) {
                if (!valid_subsample_row(i)) continue;

                uint16_t* __restrict assignment_row = assignment.get_row(i);
                float* __restrict min_dist_row = min_dists.get_row(i);
                for (int j = x_lo; j < x_hi; j++) {
                    float dr = quad_image.get(i, 4 * j) - cluster_r;
                    float dg = quad_image.get(i, 4 * j + 1) - cluster_g;
                    float db = quad_image.get(i, 4 * j + 2) - cluster_b;
                    float dy = coef * (i - cluster_y), dx = coef * (j - cluster_x);

                    float distance;
                    if (use_manhattan) {
                        distance = std::fabs(dr) + std::fabs(dg) + std::fabs(db) + std::fabs(dx) + std::fabs(dy);
                    } else {
                        distance = dr*dr + dg*dg + db*db + dx*dx + dy*dy;
                    }
                    if (min_dist_row[j] > distance) {
                        min_dist_row[j] = distance;
                        assignment_row[j] = cluster_no;
                    }
                }
            }
        }
    }

    template class BaseContext<float>;
    template class BaseContext<double>;
    template class BaseContext<uint16_t>;
    template class BaseContext<uint32_t>;
};
