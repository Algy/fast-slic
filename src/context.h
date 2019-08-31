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
#include "preemptive.h"
#include "recorder.h"

typedef std::chrono::high_resolution_clock Clock;

namespace fslic {
    template <typename DistType>
    class BaseContext {
    public:
        int16_t subsample_stride_config = 3;
        int num_threads = -1;
        float compactness = 20;
        float min_size_factor = 0.1;
        bool convert_to_lab = false;

        bool preemptive = false;
        float preemptive_thres = 0.01;

        bool manhattan_spatial_dist = true;
        bool debug_mode = false;
    protected:
        int H, W, K;
        const uint8_t* image;
        Cluster* clusters;
        int16_t S;
    protected:
        int16_t subsample_rem;
        int16_t subsample_stride;

    protected:
        int color_shift;
    protected:
        simd_helper::AlignedArray<uint8_t> quad_image;
        simd_helper::AlignedArray<uint16_t> assignment;
        simd_helper::AlignedArray<DistType> min_dists;
        simd_helper::AlignedArray<DistType> spatial_dist_patch;

        PreemptiveGrid preemptive_grid;
        Recorder<DistType> recorder;
    public:
        std::string last_timing_report;
    public:
        BaseContext(int H, int W, int K, const uint8_t* image, Cluster *clusters)
            : H(H), W(W), K(K), image(image), clusters(clusters), S(sqrt(H * W / K)),
              quad_image(H, 4 * W, S, S, 4 * S, 4 * S),
              assignment(H, W, S, S, S, S),
              min_dists(H, W, S, S, S, S),
              spatial_dist_patch(2 * S + 1, 2 * S + 1),
              preemptive_grid(H, W, K, S),
              recorder(H, W, K) {};
        virtual ~BaseContext();
    public:
        void initialize_clusters();
        void initialize_state();
        void enforce_connectivity(uint16_t *assignment);
        bool parallelism_supported();
        void iterate(uint16_t *assignment, int max_iter);
        std::string get_timing_report() { return last_timing_report; };
        std::string get_recorder_report() { return recorder.get_report(); };
    private:
        void prepare_spatial();
        void assign();
        void update();
    protected:
        void full_assign();
        template <typename T>
        inline T fit_to_stride(T value) {
            T plus_rem = subsample_rem - value % subsample_stride;
            if (plus_rem < 0) plus_rem += subsample_stride;
            return value + plus_rem;
        }

        inline bool valid_subsample_row(int i) {
            return i % subsample_stride == subsample_rem;
        }
    protected:
        virtual void before_iteration() {};
        virtual void after_update() {};
        virtual void set_spatial_patch();
        virtual void assign_clusters(const Cluster **target_clusters, int size);
        virtual bool centroid_quantization_enabled();
    };

    class ContextRealDist : public BaseContext<float> {
    public:
        using BaseContext<float>::BaseContext;
    };

    class ContextRealDistL2 : public ContextRealDist {
    public:
        using ContextRealDist::ContextRealDist;
    protected:
        virtual void set_spatial_patch();
        virtual void assign_clusters(const Cluster **target_clusters, int size);
    };

    class ContextRealDistNoQ : public ContextRealDist {
    public:
        using ContextRealDist::ContextRealDist;
        bool float_color = true;
    protected:
        virtual void assign_clusters(const Cluster **target_clusters, int size);
        virtual bool centroid_quantization_enabled();
    private:
        template<bool use_manhattan>
        void assign_clusters_proto(const Cluster **target_clusters, int size);
    protected:
        virtual void before_iteration();
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
