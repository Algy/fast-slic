#ifndef _FAST_SLIC_CONTEXT_H
#define _FAST_SLIC_CONTEXT_H

#include <vector>
#include <chrono>
#include <cassert>
#include <cstring>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include "simd-helper.hpp"
#include "fast-slic-common.h"

typedef std::chrono::high_resolution_clock Clock;

namespace fslic {
    template <typename DistType>
    class BaseContext {
    public:
        int16_t subsample_stride_config = 3;
        int num_threads = 0;
        float compactness = 20;
        float min_size_factor = 0.1;
        int color_shift = 4;
    protected:
        int H, W, K;
        int16_t S;
        Cluster* clusters;
        const uint8_t* image;
    protected:
        int16_t subsample_rem;
        int16_t subsample_stride;

        DistType* spatial_dist_patch = nullptr;
        uint16_t patch_memory_width;
        uint16_t patch_virtual_width;
        uint16_t patch_height;

        DistType* spatial_normalize_cache = nullptr;
    protected:
        uint8_t* aligned_quad_image_base = nullptr;
        uint8_t* aligned_quad_image = nullptr; // copied image
        uint16_t quad_image_memory_width;
        uint16_t* aligned_assignment_base = nullptr;
        uint16_t* aligned_assignment = nullptr;
        DistType* aligned_min_dists_base = nullptr;
        DistType* aligned_min_dists = nullptr;
        int assignment_memory_width; // memory width of aligned_assignment
        int min_dist_memory_width; // memory width of aligned_min_dists;
    public:
        BaseContext(int H, int W, int K, const uint8_t* image, Cluster *clusters) : H(H), W(W), K(K), image(image), clusters(clusters), S(sqrt(H * W / K)) {};
        virtual ~BaseContext();
    public:
        template <typename T>
        inline T fit_to_stride(T value) {
            T plus_rem = subsample_rem - value % subsample_stride;
            if (plus_rem < 0) plus_rem += subsample_stride;
            return value + plus_rem;
        }

        inline bool valid_subsample_row(int i) {
            return i % subsample_stride == subsample_rem;
        }

    public:
        void initialize_clusters();
        void initialize_state();
        void enforce_connectivity(uint16_t *assignment);
        bool parallelism_supported();
        void iterate(uint16_t *assignment, int max_iter);
    private:
        void prepare_spatial();
        void assign();
        void update();
    protected:
        virtual void assign_clusters(const Cluster **target_clusters, int size);
    };

    class ContextRealDist : public BaseContext<float> {
    public:
        using BaseContext<float>::BaseContext;
    };

    class Context : public BaseContext<uint16_t> {
    public:
        using BaseContext<uint16_t>::BaseContext;
    };

    class ContextSIMD : public Context {
    public:
        using Context::Context;
    };

    class ContextBuilderImpl;
    class ContextBuilder {
    private:
        std::unique_ptr<ContextBuilderImpl> impl;
    public:
        ContextBuilder() : ContextBuilder("standard") {};
        ContextBuilder(const char* arch);
        virtual ~ContextBuilder();
        const char** supported_archs();
        bool is_supported_arch();
        const char* get_arch();
        void set_arch(const char*);
        Context* build(int H, int W, int K, const uint8_t* image, Cluster *clusters);
    };
};

#endif
