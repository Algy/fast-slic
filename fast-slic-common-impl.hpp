#include <vector>
#include <chrono>
#include <cassert>
#include <cstring>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <climits>
#include <list>
#include <atomic>
#include "simd-helper.hpp"
#include "fast-slic-common.h"
#include "cca.h"

typedef std::chrono::high_resolution_clock Clock;

#define CHARBIT 8
#ifdef _MSC_VER
#define __restrict__ __restrict
#endif

template <typename T>
static inline T my_max(T x, T y) {
    return (x > y) ? x : y;
}


template <typename T>
static inline T my_min(T x, T y) {
    return (x < y) ? x : y;
}

template <typename T>
static inline T clamp(T value, T lo, T hi) {
    return (value < lo) ? lo : ((value > hi)? hi : value);
}


template <typename T>
static T fast_abs(T n)
{
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


class BaseContext {
public:
    int H, W, K;
    int16_t S;
    const char* algorithm;
    float compactness;
    float min_size_factor = 0.1;
    int16_t subsample_stride = 3;
    int16_t subsample_rem = 1;
    Cluster* __restrict__ clusters;
    const uint8_t* __restrict__ image = nullptr;
    uint16_t* __restrict__ spatial_dist_patch = nullptr;
    uint16_t* __restrict__ spatial_normalize_cache = nullptr;
    uint16_t* __restrict__ assignment = nullptr;

public:
    virtual ~BaseContext() {
        if (spatial_dist_patch) {
            simd_helper::free_aligned_array(spatial_dist_patch);
        }
        if (spatial_normalize_cache) {
            delete [] spatial_normalize_cache;
        }
    }

    virtual void prepare_spatial() {
        if (spatial_normalize_cache) delete [] spatial_normalize_cache;
        spatial_normalize_cache = new uint16_t[2 * S + 2];
        for (int x = 0; x < 2 * S + 2; x++) {
            // rescale distance [0, 1] to [0, 25.5] (color-scale).
            uint16_t val = compactness * (float)x / (float)S;
            spatial_normalize_cache[x] = val;
        }

        const uint16_t patch_height = 2 * S + 1, patch_virtual_width = 2 * S + 1;
        const uint16_t patch_memory_width = simd_helper::align_to_next(patch_virtual_width);

        if (spatial_dist_patch) simd_helper::free_aligned_array(spatial_dist_patch);
        spatial_dist_patch = simd_helper::alloc_aligned_array<uint16_t>(patch_height * patch_memory_width);
        uint16_t row_first_manhattan = 2 * S;
        // first half lines
        for (int i = 0; i < S; i++) {
            uint16_t current_manhattan = row_first_manhattan--;
            // first half columns
            for (int j = 0; j < S; j++) {
                uint16_t val = spatial_normalize_cache[current_manhattan--];
                spatial_dist_patch[i * patch_memory_width + j] = val;
            }
            // half columns next to the first columns
            for (int j = S; j <= 2 * S; j++) {
                uint16_t val = spatial_normalize_cache[current_manhattan++];
                spatial_dist_patch[i * patch_memory_width + j] = val;
            }
        }

        // next half lines
        for (int i = S; i <= 2 * S; i++) {
            uint16_t current_manhattan = row_first_manhattan++;
            // first half columns
            for (int j = 0; j < S; j++) {
                uint16_t val = spatial_normalize_cache[current_manhattan--];
                spatial_dist_patch[i * patch_memory_width + j] = val;
            }
            // half columns next to the first columns
            for (int j = S; j <= 2 * S; j++) {
                uint16_t val = spatial_normalize_cache[current_manhattan++];
                spatial_dist_patch[i * patch_memory_width + j] = val;
            }
        }
    }

    template <typename T>
    inline T fit_to_stride(T value) {
        T plus_rem = subsample_rem - value % subsample_stride;
        if (plus_rem < 0) plus_rem += subsample_stride;
        return value + plus_rem;
    }

    inline bool valid_subsample_row(int i) {
        return i % subsample_stride == subsample_rem;
    }

    void enforce_connectivity() {
        int thres = (int)round((double)(S * S) * (double)min_size_factor);
        if (K <= 0 || H <= 0 || W <= 0) return;
        cca::ConnectivityEnforcer ce(assignment, H, W, K, thres);
        ce.execute(assignment);
    }
};


static void do_fast_slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters) {
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
