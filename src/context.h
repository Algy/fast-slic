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
#include "tile.h"


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
        std::vector<uint8_t> image;
        const uint8_t* orig_image;
        Cluster* clusters;
        int16_t S;
        TileSet<DistType, 8> tile_set;
        SpatialDistancePatch<DistType, 8> dist_patch;
    protected:
        int16_t subsample_rem;
        int16_t subsample_stride;

    protected:
        int color_shift;
    protected:
        PreemptiveGrid preemptive_grid;
        Recorder<DistType> recorder;
    public:
        std::string last_timing_report;
    public:
        BaseContext(int H, int W, int K, const uint8_t* image, Cluster *clusters);
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
        virtual void assign_clusters(int tile_no);
        virtual bool centroid_quantization_enabled();
    };

    class ContextRealDist : public BaseContext<float> {
    public:
        using BaseContext<float>::BaseContext;
    };

    class ContextRealDistL2 : public ContextRealDist {
    public:
        using ContextRealDist::ContextRealDist;
    };

    class ContextRealDistNoQ : public ContextRealDist {
    public:
        using ContextRealDist::ContextRealDist;
        bool float_color = true;
    };

    class Context : public BaseContext<uint8_t> {
    public:
        using BaseContext<uint8_t>::BaseContext;
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
