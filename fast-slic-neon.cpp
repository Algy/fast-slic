#include "fast-slic-neon.h"
#include "context.h"

extern "C" {
    void fast_slic_initialize_clusters_neon(int H, int W, int K, const uint8_t* image, Cluster *clusters) {
        fslic::ContextBuilder builder("arm/neon");
        std::unique_ptr<fslic::Context> ctx { builder.build(H, W, K, image, clusters) };
        ctx->initialize_clusters();
    }

    void fast_slic_iterate_neon(int H, int W, int K, float compactness, float min_size_factor, uint8_t subsample_stride, int max_iter, const uint8_t *__restrict__ image, Cluster *__restrict__ clusters, uint16_t* __restrict__ assignment) {
        fslic::ContextBuilder builder("arm/neon");
        std::unique_ptr<fslic::Context> ctx { builder.build(H, W, K, image, clusters) };
        ctx->compactness = compactness;
        ctx->min_size_factor = min_size_factor;
        ctx->subsample_stride_config = subsample_stride;
        ctx->initialize_state();
        ctx->iterate(assignment, max_iter);
    }

    int fast_slic_supports_neon() {
        fslic::ContextBuilder builder("arm/neon");
        return builder.is_supported_arch();
    }
}


#ifdef PROTOTYPE_MAIN_DEMO

#ifndef USE_NEON
#error "Compile it with flag USE_NEON"
#endif

#include <cstdlib>
#include <ctime>
#include <string>
#include <chrono>
#include <fstream>
#include <memory>
typedef std::chrono::high_resolution_clock Clock;
int main(int argc, char** argv) {
    int K = 100;
    int compactness = 5;
    int max_iter = 2;
    int subsample_stride = 6;
    try {
        if (argc > 1) {
            K = std::stoi(std::string(argv[1]));
        }
        if (argc > 2) {
            compactness = std::stoi(std::string(argv[2]));
        }
        if (argc > 3) {
            max_iter = std::stoi(std::string(argv[3]));
        }
        if (argc > 4) {
            subsample_stride = std::stoi(std::string(argv[4]));
        }
    } catch (...) {
        std::cerr << "slic num_components compactness max_iter subsample_stride" << std::endl;
        return 2;
    }

    int H = 480;
    int W = 640;
    srand(time(nullptr));

    Cluster clusters[K];
    std::unique_ptr<uint8_t[]> image { new uint8_t[H * W * 3] };
    std::unique_ptr<uint16_t[]> assignment { new uint16_t[H * W] };

    /*
    std::ifstream inputf("/tmp/a.txt");
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int r, g, b;
            inputf >> r >> g >> b;
            image.get()[3 * W * i + 3 * j] = r;
            image.get()[3 * W * i + 3 * j + 1] = g;
            image.get()[3 * W * i + 3 * j + 2] = b;
        }
    }
    */
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int r, g, b;
            r = (int)(rand() * 255);
            g = (int)(rand() * 255);
            b = (int)(rand() * 255);
            image.get()[3 * W * i + 3 * j] = r;
            image.get()[3 * W * i + 3 * j + 1] = g;
            image.get()[3 * W * i + 3 * j + 2] = b;
        }
    }

    auto t1 = Clock::now();
    fast_slic_initialize_clusters_neon(H, W, K, image.get(), clusters);
    fast_slic_iterate_neon(H, W, K, compactness, 0.1, subsample_stride, max_iter, image.get(), clusters, assignment.get());

    auto t2 = Clock::now();
    // 6 times faster than skimage.segmentation.slic
    std::cerr << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";

    {
        std::ofstream outputf("/tmp/b.output.txt");
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                outputf << (short)assignment.get()[W * i + j] << " ";
            }
            outputf << std::endl;
        }
    }
    {
        std::ofstream outputf("/tmp/b.clusters.txt");
        for (int k = 0; k < K; k++) {
            outputf << clusters[k].y << " " << clusters[k].x << " " << clusters[k].num_members << std::endl;
        }
    }
    return 0;
}
#endif
