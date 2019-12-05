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

template <typename DistType, int TileWidth>
class SpatialDistancePatch {
private:
    int S;
    int width;
    std::vector<DistType> data_y;
    std::vector<DistType> data_x;
public:
    SpatialDistancePatch(int S) :
            S(S),
            width((2 * S + TileWidth) * TileWidth * TileWidth),
            data_y(simd_helper::align_to_next(width)),
            data_x(simd_helper::align_to_next(width)) {};
    void set(float compactness) {
        float coef = (compactness / S) * 1.41f;

        for (int n = -S; n < S + TileWidth; n++) {
            DistType* __restrict y_arr = at_y(n);
            DistType* __restrict x_arr = at_x(n);

            for (int i = 0; i < TileWidth; i++) {
                DistType y_val = (DistType)(coef * fast_abs(i - n));
                for (int j = 0; j < TileWidth; j++) {
                    y_arr[i * TileWidth + j] = y_val;
                    x_arr[i * TileWidth + j] = (DistType)(coef * fast_abs(j - n));
                }
            }
        }
    }

    DistType* __restrict at_y(int offset) {
        return (DistType * __restrict) &data_y[(TileWidth * TileWidth) * (offset + S)];
    }

    DistType* __restrict at_x(int offset) {
        offset = clamp(offset, -S, S + TileWidth - 1);
        return (DistType * __restrict) &data_x[(TileWidth * TileWidth) * (offset + S)];
    }
    bool in_range(int offset) {
        return offset >= -S && offset < S + TileWidth;
    }
};


template <typename DistType, int TileWidth = 8>
class TileSet {
private:
    int H, W, S, num_rows, num_cols;
    int num_tiles, tile_size, tile_memory_size;
    std::vector<LimitedSlots<uint16_t, 16>> neighbor_cluster_nos;
    std::vector<Tile> tiles;

    std::vector<uint8_t> tile_color[3];
    std::vector<DistType> tile_min_dists;
    std::vector<uint8_t> tile_min_neighbor_indices;
public:
    TileSet(int H, int W, int S) : H(H), W(W), S(S),
            num_rows(ceil_int(H, TileWidth)), num_cols(ceil_int(W, TileWidth)),
            num_tiles(num_rows*num_cols), tile_size(TileWidth*TileWidth), tile_memory_size(simd_helper::align_to_next(tile_size)),
            neighbor_cluster_nos(num_tiles), tiles(num_tiles) {
        tile_min_dists.resize(num_tiles*tile_memory_size, std::numeric_limits<DistType>::max());
        tile_min_neighbor_indices.resize(num_tiles*tile_memory_size, 0xFF);
        for (int i = 0; i < 3; i++) {
            tile_color[i].resize(num_tiles*tile_memory_size);
        }

        for (int ty = 0; ty < num_rows; ty++) {
            int sy = ty * TileWidth;
            int ey = my_min(H, sy + TileWidth);
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
    int get_num_tiles() const { return num_tiles; };
    int get_tile_memory_size() const { return tile_memory_size; };


    const uint8_t* get_color_plane(int tile_no, int plane_no) const {
        return &tile_color[plane_no][tile_memory_size * tile_no];
    }

    void set_clusters(const Cluster* clusters, int K) {
        for (int k = 0; k < K; k++) {
            const int cy = clusters[k].y, cx = clusters[k].x;
            const int sty = my_max(cy - S, 0) / TileWidth, ety = my_min(cy + S, H - 1) / TileWidth;
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
        for (int tile_no = 0; tile_no < num_tiles; tile_no++) {
            const Tile &tile = tiles[tile_no];
            uint8_t *r_plane = &tile_color[0][tile_memory_size * tile_no];
            uint8_t *g_plane = &tile_color[1][tile_memory_size * tile_no];
            uint8_t *b_plane = &tile_color[2][tile_memory_size * tile_no];

            for (int i = tile.sy; i < tile.ey; i++) {
                for (int j = tile.sx; j < tile.ex; j++) {
                    const uint8_t r = image[3 * (W * i + j)],
                            g = image[3 * (W * i + j) + 1],
                            b = image[3 * (W * i + j) + 2];
                    r_plane[(i - tile.sy) * memory_width + (j - tile.sx)] = r;
                    g_plane[(i - tile.sy) * memory_width + (j - tile.sx)] = g;
                    b_plane[(i - tile.sy) * memory_width + (j - tile.sx)] = b;
                }
            }
        }
    }

    void initialize_dists() {
        reset_dists();
        std::fill(
                tile_min_neighbor_indices.begin(),
                tile_min_neighbor_indices.end(),
                0xFF
        );
    }

    void reset_dists() {
        std::fill(
                tile_min_dists.begin(),
                tile_min_dists.end(),
                std::numeric_limits<DistType>::max()
        );
    }

    DistType* get_tile_min_dists(int tile_no) {
        return &tile_min_dists[tile_memory_size * tile_no];
    }

    uint8_t* get_tile_min_neighbor_indices(int tile_no) {
        return &tile_min_neighbor_indices[tile_memory_size * tile_no];
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
