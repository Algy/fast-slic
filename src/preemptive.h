#include <algorithm>
#include <vector>
#include <cstdint>
#include "fast-slic-common.h"
#include "simd-helper.hpp"
#include "parallel.h"

struct PreemptiveTile {
    int sy, sx, ey, ex;
};

class PreemptiveGrid {
private:
    bool enabled;
    int H;
    int W;
    int S;
    int CW;
    int CH;
    float thres;
    int stride;
    std::vector<int> num_changes;
    std::vector<bool> is_updatable;
    std::vector<bool> is_active;
    bool b_all_active;
    simd_helper::AlignedArray<uint16_t> old_assignment;
public:
    PreemptiveGrid(int H, int W, int S) : H(H), W(W), S(S), stride(1) {
        CW = ceil_int(W, S);
        CH = ceil_int(H, S);
        num_changes.resize(CH * CW, 0);
        is_updatable.resize(CH * CW, true);
        is_active.resize(CH * CW, true);
        b_all_active = true;
    };

    void initialize(bool enabled, float thres, int stride) {
        this->enabled = enabled;
        this->thres = thres;
        this->stride = stride;
        std::fill(num_changes.begin(), num_changes.end(), 0);
        std::fill(is_updatable.begin(), is_updatable.end(), true);
        std::fill(is_active.begin(), is_active.end(), true);
        b_all_active = true;
    }

    void finalize() {
        std::fill(is_updatable.begin(), is_updatable.end(), true);
        std::fill(is_active.begin(), is_active.end(), true);
        b_all_active = true;
    }

    bool all_active() {
        return !enabled || b_all_active;
    }

    bool is_active_cluster(const Cluster& cluster) {
        return !enabled || is_active[CW * (cluster.y / S) + (cluster.x / S)];
    }

    bool is_updatable_cluster(const Cluster& cluster) {
        return !enabled || is_updatable[CW * (cluster.y / S) + (cluster.x / S)];
    }

    std::vector<PreemptiveTile> get_active_tiles() const {
        std::vector<PreemptiveTile> result;
        for (int ci = 0; ci < CH; ci++) {
            for (int cj = 0; cj < CW; cj++) {
                int cell_index = CW * ci + cj;
                if (!is_active[cell_index]) continue;
                PreemptiveTile tile;
                tile.sy = ci * S;
                tile.sx = cj * S;
                tile.ey = my_min(tile.sy + S, H);
                tile.ex = my_min(tile.sx + S, W);
                result.push_back(tile);
            }
        }
        return result;
    }

    void set_old_assignment(const simd_helper::AlignedArray<uint16_t> &assignment) {
        if (!enabled) return;
        this->old_assignment = assignment;
        std::fill(num_changes.begin(), num_changes.end(), 0);
    }

    void set_new_assignment(const simd_helper::AlignedArray<uint16_t> &assignment) {
        if (!enabled) return;
        std::fill(num_changes.begin(), num_changes.end(), 0);

        #pragma omp parallel for num_threads(fsparallel::nth())
        for (int ci = 0; ci < CH; ci++) {
            for (int cj = 0; cj < CW; cj++) {
                if (!is_active[CW * ci + cj]) continue;
                int cell_index = CW * ci + cj;
                int ei = my_min((ci + 1) * S, H);
                for (int i = ci * S; i < ei; i++) {
                    int ej = my_min((cj + 1) * S, W);
                    for (int j = cj * S; j < ej; j++) {
                        uint16_t old_label = old_assignment.get(i, j);
                        uint16_t new_label = assignment.get(i, j);
                        if (new_label != old_label)
                            num_changes[cell_index]++;
                    }
                }
            }
        }

        int thres_by_num_cells[10];
        for (int i = 0; i <= 9; i++) {
            thres_by_num_cells[i] = (int)(i * thres * S * S / stride);
        }
        const int dir[3] = {-1, 0, 1};
        std::fill(is_active.begin(), is_active.end(), false);
        for (int ci = 0; ci < CH; ci++) {
            for (int cj = 0; cj < CW; cj++) {
                int num_cells = 0;
                int curr_num_changes = 0;
                for (int di : dir) {
                    int curr_ci = ci + di;
                    if (curr_ci < 0 || curr_ci >= CH) continue;
                    for (int dj : dir) {
                        int curr_cj = cj + dj;
                        if (curr_cj < 0 || curr_cj >= CW) continue;
                        num_cells++;
                        curr_num_changes += num_changes[CW * curr_ci + curr_cj];
                    }
                }
                bool updatable = curr_num_changes > thres_by_num_cells[num_cells];
                is_updatable[ci * CW + cj] = updatable;

                if (updatable) {
                    for (int di : dir) {
                        int curr_ci = ci + di;
                        if (curr_ci < 0 || curr_ci >= CH) continue;
                        for (int dj : dir) {
                            int curr_cj = cj + dj;
                            if (curr_cj < 0 || curr_cj >= CW) continue;
                            is_active[CW * curr_ci + curr_cj] = true;
                        }
                    }
                }
            }
        }

        b_all_active = true;
        for (bool active: is_active) {
            if (!active) {
                b_all_active = false;
                break;
            }
        }
    }
};
