#include <algorithm>
#include <vector>
#include <cstdint>
#include "fast-slic-common.h"
#include "simd-helper.hpp"
#include "parallel.h"
#include "timer.h"

struct PreemptiveTile {
    int sy, sx, ey, ex;
};

class PreemptiveGrid {
private:
    bool enabled;
    int H;
    int W;
    int K;
    int S;
    int CW;
    int CH;
    float thres;
    int stride;
    std::vector<Cluster> old_clusters;
    std::vector<std::vector<uint16_t>> cluster_grid;
    std::vector<int> active_grid;
    bool b_all_active;

    std::vector<int> y_to_cell_y;
    std::vector<int> x_to_cell_x;

    int cooldown = 2;
public:
    PreemptiveGrid(int H, int W, int K, int S) : H(H), W(W), K(K), S(S), stride(1),
            y_to_cell_y(H), x_to_cell_x(W) {
        CW = ceil_int(W, 2 * S);
        CH = ceil_int(H, 2 * S);
        old_clusters.resize(K);
        cluster_grid.resize(CH * CW);
        active_grid.resize(CH * CW);
        b_all_active = true;

        for (int ci = 0; ci < CH; ci++) {
            int i_start = ci * 2 * S;
            int i_end = my_min(i_start + 2 * S, H);
            for (int i = i_start; i < i_end; i++) {
                y_to_cell_y[i] = ci;
            }
        }
        for (int cj = 0; cj < CW; cj++) {
            int j_start = cj * 2 * S;
            int j_end = my_min(j_start + 2 * S, W);
            for (int j = j_start; j < j_end; j++) {
                x_to_cell_x[j] = cj;
            }
        }
    };

    void initialize(Cluster* clusters, bool enabled, float thres, int stride) {
        this->enabled = enabled;
        this->thres = thres;
        this->stride = stride;
        b_all_active = true;
        for (int k = 0; k < K; k++) {
            clusters[k].is_updatable = cooldown;
        }
    }

    void finalize(Cluster* clusters) {
        b_all_active = true;
        for (int k = 0; k < K; k++) {
            clusters[k].is_active = 1;
        }
    }

    bool all_active() {
        return !enabled || b_all_active;
    }

    std::vector<PreemptiveTile> get_active_tiles() const {
        std::vector<PreemptiveTile> result;
        if (!b_all_active) {
            for (int ci = 0; ci < CH; ci++) {
                for (int cj = 0; cj < CW; cj++) {
                    int cell_index = CW * ci + cj;
                    if (!active_grid[cell_index]) continue;
                    PreemptiveTile tile;
                    tile.sy = ci * 2 * S;
                    tile.sx = cj * 2 * S;
                    tile.ey = my_min(tile.sy + 2 * S, H);
                    tile.ex = my_min(tile.sx + 2 * S, W);
                    result.push_back(tile);
                }
            }
        } else {
            PreemptiveTile tile;
            tile.sx = tile.sy = 0;
            tile.ey = H;
            tile.ex = W;
            result.push_back(tile);
        }
        return result;
    }

    void set_old_clusters(const Cluster* clusters) {
        if (!enabled) return;
        std::copy(clusters, clusters + K, old_clusters.begin());
    }

    int& get_active_cell(int y, int x) {
        return active_grid[CW * y_to_cell_y[y] + x_to_cell_x[x]];
    }

    void set_new_clusters(Cluster* clusters) {
        fstimer::Scope s("set_new_clusters");
        if (!enabled) return;

        for (auto &list : cluster_grid) list.clear();
        for (int k = 0; k < K; k++) {
            cluster_grid[CW * y_to_cell_y[(int)clusters[k].y] + x_to_cell_x[(int)clusters[k].x]].push_back(k);
            clusters[k].is_active = 0;
        }

        std::fill(active_grid.begin(), active_grid.end(), 0);

        float l1_thres = my_max(roundf(2 * S * thres), 1.0f);
        const int dir[3] = {-1, 0, 1};
        int num_active = 0;
        #pragma omp parallel num_threads(fsparallel::nth())
        {
            #pragma omp for
            for (int k = 0; k < K; k++) {
                if (!clusters[k].is_updatable) continue;
                float l1_diff = abs(old_clusters[k].x - clusters[k].x) + abs(old_clusters[k].y - clusters[k].y);
                if (l1_diff < l1_thres) {
                    clusters[k].is_updatable--;
                } else {
                    clusters[k].is_updatable = cooldown;
                }
            }

            #pragma omp for
            for (int k = 0; k < K; k++) {
                const Cluster* cluster = &clusters[k];
                if (!cluster->is_updatable) continue;

                int y = cluster->y, x = cluster->x;
                int cy = y_to_cell_y[y], cx = x_to_cell_x[x];
                bool any_active = false;
                for (int dy : dir) {
                    int ny = cy + dy;
                    if (!(ny >= 0 && ny < CH)) continue;
                    for (int dx : dir) {
                        int nx = cx + dx;
                        if (!(nx >= 0 && nx < CW)) continue;
                        for (uint16_t neighbor_no : cluster_grid[CW * ny + nx]) {
                            Cluster* neighbor = &clusters[neighbor_no];
                            int neighbor_y = neighbor->y, neighbor_x = neighbor->x;
                            if (fast_abs(neighbor_y - y) <= 2 * S &&
                                    fast_abs(neighbor_x - x) <= 2 * S) {
                                neighbor->is_active = 1;
                                get_active_cell(neighbor_y, neighbor_x) = 1;
                            }
                        }
                    }
                }
            }

            #pragma omp for
            for (int k = 0; k < K; k++) {
                #pragma omp atomic
                num_active += (int)clusters[k].is_active;
            }

            #pragma omp single
            b_all_active = num_active == K;
        }
    }
};
