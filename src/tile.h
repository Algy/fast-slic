//
// Created by algy on 19. 11. 23..
//

#ifndef FAST_SLIC_TILE_H
#define FAST_SLIC_TILE_H

#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>
#include "parallel.h"
#include "simd-helper.hpp"
#include "fast-slic-common.h"

template <class T, unsigned char limit>
class LimitedSlots {
private:
    uint8_t n;
    T cells[limit];
public:
    int get_limit() const { return limit; };
    void initialize_value(T t) { std::fill(cells, cells + limit, t); };
    int size() const { return n; };
    void push_back(T t) {
        if (n < limit) {
            cells[n++] = t;
        }
    }
    T& operator[](int i) { return cells[i]; };
    const T& operator[](int i) const { return cells[i]; };
};


struct Tile {
    int sy, sx, ey, ex;

    int width() const { return ex - sx; };
    int height() const { return ey - sy; };
};

template <typename DistType, int TileWidth = 32, int TileHeight = 1>
class SpatialDistancePatch {
private:
    int S;
    int width;
    simd_helper::AlignedArray<DistType> data;
public:
    SpatialDistancePatch(int S) : S(S), data(2 * (S + TileHeight) + 1, 2 * (S + TileWidth)  + 1) {};
    void set(float compactness, int color_shift) {
        float coef = compactness / S;
        coef *= (1 << color_shift);

        for (int i = 0; i < data.get_height(); i++) {
            for (int j = 0; j < data.get_width(); j++) {
                data.get(i, j) = std::numeric_limits<DistType>::max();
            }
        }

        for (int i = -S; i <= S; i++) {
            for (int j = -S; j <= S; j++) {
                data.get(i + get_center_y(), j + get_center_x()) = (DistType)(coef * (fast_abs(i) + fast_abs(j)));
            }
        }
    }
    const DistType* at(int cy, int cx) const {
        return &data.get(cy + get_center_y(), cx + get_center_x());
    }

    bool y_in_range(int y) { return y >= -S && y <= S; };
    bool x_in_range(int x) { return x >= -(S + TileWidth - 1) && x <= S; };
private:
    int get_center_y() const { return TileHeight + S; };
    int get_center_x() const { return TileWidth + S; };
};


template <typename DistType, int TileWidth = 32, int TileHeight = 1>
class TileSet {
private:
    int H, W, S, num_rows, num_cols;
    int num_tiles;
    int plane_memory_size;
    std::vector<LimitedSlots<uint16_t, 16>> neighbor_cluster_nos;
    std::vector<Tile> tiles;
    std::vector<uint8_t> r_plane, g_plane, b_plane;
    std::vector<DistType> tile_min_dists;
    std::vector<uint8_t> tile_min_neighbor_indices;
public:
    TileSet(int H, int W, int S) : H(H), W(W), S(S),
            num_rows(ceil_int(H, TileHeight)), num_cols(ceil_int(W, TileWidth)),
            num_tiles(num_rows*num_cols),
            plane_memory_size(get_tile_memory_size() * num_tiles),
            neighbor_cluster_nos(num_tiles), tiles(num_tiles),
            r_plane(plane_memory_size), g_plane(plane_memory_size), b_plane(plane_memory_size),
            tile_min_dists(plane_memory_size, std::numeric_limits<DistType>::max()),
            tile_min_neighbor_indices(plane_memory_size, 0xFF) {
        for (int ty = 0; ty < num_rows; ty++) {
            int sy = ty * TileHeight;
            int ey = my_min(H, sy + TileHeight);
            for (int tx = 0; tx < num_cols; tx++) {
                int sx = tx * TileWidth;
                int ex = my_min(W, sx + TileWidth);
                int tile_no = num_cols * ty + tx;
                tiles[tile_no].sy = sy;
                tiles[tile_no].sx = sx;
                tiles[tile_no].ey = ey;
                tiles[tile_no].ex = ex;
            }
        }
    }
    inline int get_tile_memory_size() const { return TileWidth*TileHeight; };
    inline int get_tile_no(int row, int col) const { return row * num_cols + col; };

    uint8_t* get_r_plane(int tile_no) { return &r_plane[tile_no * get_tile_memory_size()]; };
    uint8_t* get_g_plane(int tile_no) { return &g_plane[tile_no * get_tile_memory_size()]; };
    uint8_t* get_b_plane(int tile_no) { return &b_plane[tile_no * get_tile_memory_size()]; };

    void set_clusters(const Cluster* clusters, int K) {
        #pragma omp parallel for num_threads(fsparallel::nth())
        for (int k = 0; k < K; k++) {
            const int cy = clusters[k].y, cx = clusters[k].x;
            const int sty = my_max(cy - S, 0) / TileHeight, ety = my_min(cy + S, H - 1) / TileHeight;
            const int stx = my_max(cx - S, 0) / TileWidth, etx = my_min(cx + S, W - 1) / TileWidth;

            for (int ty = sty; ty <= ety; ty++) {
                for (int tx = stx; tx <= etx; tx++) {
                    neighbor_cluster_nos[ty * num_cols + tx].push_back(k);
                }
            }
        }
    }

    void set_image(const uint8_t* image) {
        int memory_width = get_tile_memory_width();

        #pragma omp parallel for num_threads(fsparallel::nth())
        for (int tile_no = 0; tile_no < num_tiles; tile_no++) {
            const Tile &tile = tiles[tile_no];
            uint8_t* rs = get_r_plane(tile_no), *gs = get_g_plane(tile_no), *bs = get_b_plane(tile_no);

            for (int i = tile.sy; i < tile.ey; i++) {
                for (int j = tile.sx; j < tile.ex; j++) {
                    const uint8_t r = image[3 * (W * i + j)],
                            g = image[3 * (W * i + j) + 1],
                            b = image[3 * (W * i + j) + 2];
                    int ix = (i - tile.sy) * memory_width + (j - tile.sx);
                    rs[ix] = r;
                    gs[ix] = g;
                    bs[ix] = b;
                }
            }
        }
    }

    void reset_dists() {
        std::fill(
                tile_min_dists.begin(),
                tile_min_dists.end(),
                std::numeric_limits<DistType>::max()
        );
        std::fill(
                tile_min_neighbor_indices.begin(),
                tile_min_neighbor_indices.end(),
                0xFF
        );
    }

    DistType* get_tile_min_dists(int tile_no) {
        return &tile_min_dists[get_tile_memory_size() * tile_no];
    }

    uint8_t* get_tile_min_neighbor_indices(int tile_no) {
        return &tile_min_neighbor_indices[get_tile_memory_size() * tile_no];
    }

    const LimitedSlots<uint16_t, 16>& get_neighbor_cluster_nos(int tile_no) const {
        return neighbor_cluster_nos[tile_no];
    }

    const Tile& get_tile(int tile_no) const { return tiles[tile_no]; };
    const int get_tile_memory_width() { return TileWidth; };

    int get_num_rows() const { return num_rows; };
    int get_num_cols() const { return num_cols; };

    void assign_back(uint16_t *assignment) {
        #pragma omp parallel for num_threads(fsparallel::nth())
        for (int tile_no = 0; tile_no < num_tiles; tile_no++) {
            const Tile &tile = get_tile(tile_no);
            uint8_t* min_neighbor_indices = get_tile_min_neighbor_indices(tile_no);
            auto &neighbors = get_neighbor_cluster_nos(tile_no);
            for (int i = tile.sy; i < tile.ey; i++) {
                for (int j = tile.sx; j < tile.ex; j++) {
                    int index = (i - tile.sy) * get_tile_memory_width() + (j - tile.sx);
                    assignment[W * i + j] = neighbors[min_neighbor_indices[index]];
                }
            }
        }
    }
};

#endif //FAST_SLIC_TILE_H
