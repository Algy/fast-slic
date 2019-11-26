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

template <typename DistType>
class TileSet {
private:
    int H, W, S, TH, TW;
    int num_tiles, tile_size, tile_memory_size;
    std::vector<LimitedSlots<uint16_t, 12>> neighbor_cluster_nos;
    std::vector<Tile> tiles;

    std::vector<uint8_t> tile_color[3];
    std::vector<DistType> tile_min_dists;
    std::vector<uint8_t> tile_min_neighbor_indices;
public:
    TileSet(int H, int W, int S) : H(H), W(W), S(S),
            TH(ceil_int(H, S)), TW(ceil_int(W, S)),
            num_tiles(TH*TW), tile_size(S*S), tile_memory_size(simd_helper::align_to_next(S*S)),
            neighbor_cluster_nos(num_tiles), tiles(num_tiles) {
        tile_min_dists.resize(num_tiles*tile_memory_size, 0xFF);
        tile_min_neighbor_indices.resize(num_tiles*tile_memory_size, 0xFF);
        for (int i = 0; i < 3; i++) {
            tile_color[i].resize(num_tiles*tile_memory_size);
        }

        for (int ty = 0; ty < TH; ty++) {
            int sy = ty * S;
            int ey = my_min(H, sy + S);
            for (int tx = 0; tx < TW; tx++) {
                int sx = tx * S;
                int ex = my_min(W, sx + S);
                int tile_no = TW * ty + tx;
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
        std::vector<LimitedSlots<uint16_t, 3>> grid_slots(num_tiles);
        for (int k = 0; k < K; k++) {
            int ty = clamp((int) (clusters[k].y / S), 0, TH - 1),
                    tx = clamp((int) (clusters[k].x / S), 0, TW - 1);
            grid_slots[ty * TW + tx].push_back(k);
        }

        for (int sty = 0; sty < TH; sty++) {
            for (int stx = 0; stx < TW; stx++) {
                auto& tile = neighbor_cluster_nos[sty * TW + stx];
                for (int dy = -1; dy <= 1; dy++) {
                    int ty = sty + dy;
                    if (ty < 0 || ty >= TH) continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        int tx = stx + dx;
                        if (tx < 0 || tx >= TW) continue;
                        auto &neighbors = grid_slots[ty * TW + tx];
                        for (int i = 0; i < neighbors.size(); i++) {
                            tile.push_back(neighbors[i]);
                        }
                    }
                }
            }
        }
    }

    void set_image(const uint8_t* image) {
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
                    r_plane[(i - tile.sy) * S + (j - tile.sx)] = r;
                    g_plane[(i - tile.sy) * S + (j - tile.sx)] = g;
                    b_plane[(i - tile.sy) * S + (j - tile.sx)] = b;
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

    const LimitedSlots<uint16_t, 12>& get_neighbor_cluster_nos(int tile_no) const {
        return neighbor_cluster_nos[tile_no];
    }

    const Tile& get_tile(int tile_no) const { return tiles[tile_no]; };
    const int get_tile_memory_width() { return S; };

    int get_num_rows() const { return TH; };
    int get_num_cols() const { return TW; };

    void assign_back(uint16_t *assignment) {
        #pragma omp parallel for num_threads(fsparallel::nth())
        for (int tile_no = 0; tile_no < num_tiles; tile_no++) {
            const Tile &tile = get_tile(tile_no);
            uint8_t* min_neighbor_indices = get_tile_min_neighbor_indices(tile_no);
            auto &neighbors = get_neighbor_cluster_nos(tile_no);
            for (int i = tile.sy; i < tile.ey; i++) {
                for (int j = tile.sx; j < tile.ex; j++) {
                    int index = (i - tile.sy) * S + (j - tile.sx);
                    assignment[W * i + j] = neighbors[min_neighbor_indices[index]];
                }
            }
        }
    }
};

#endif //FAST_SLIC_TILE_H
