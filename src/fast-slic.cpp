#include <utility>
#include "fast-slic.h"
#include "context.h"
#include "parallel.h"

extern "C" {
    static uint32_t symmetric_int_hash(uint32_t x, uint32_t y) {
        /*
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        */
        return ((x * 0x1f1f1f1f) ^ y) +  ((y * 0x1f1f1f1f) ^ x);
    }

    Connectivity* fast_slic_get_connectivity(int H, int W, int K, const uint16_t *assignment) {
        const static int max_conn = 12;
        Connectivity* conn = new Connectivity();
        conn->num_nodes = K;
        conn->num_neighbors = new int[K];
        conn->neighbors = new uint32_t*[K];
        std::fill_n(conn->num_neighbors, K, 0);
        std::fill_n(conn->neighbors, K, nullptr);

        uint32_t *hashtable = new uint32_t[K];
        std::fill_n(hashtable, K, 0);

        for (int i = 0; i < K; i++) {
            conn->neighbors[i] = new uint32_t[max_conn];
        }

        for (int i = 0; i < H - 1; i++) {
            for (int j = 0; j < W - 1; j++) {
                int base_index = W * i + j;
                const uint32_t source = assignment[base_index];
                int num_source_neighbors = conn->num_neighbors[source];
                if (source >= (uint32_t)K) continue;
                uint32_t* source_neighbors = conn->neighbors[source];

#               define CONNECTIVITY_SEARCH(expr) do { \
                    int target_index = expr; \
                    const uint32_t target = assignment[target_index];\
                    if (target >= (uint32_t)K || source == target) continue; \
                    const int num_target_neighbors = conn->num_neighbors[target]; \
                    if (num_source_neighbors >= max_conn || num_target_neighbors >= max_conn) continue; \
                    uint32_t* target_neighbors = conn->neighbors[target]; \
                    int hash_idx = symmetric_int_hash(source, target) % (K * 32); \
                    if (hashtable[hash_idx / 32] & (1 << (hash_idx % 32))) { \
                        bool exists = false; \
                        for (int t = 0; t < num_source_neighbors; t++) { \
                            if (source_neighbors[t] == target) { \
                                exists = true; \
                                break; \
                            } \
                        } \
                        if (exists) continue; \
                        for (int t = 0; t < num_target_neighbors; t++) { \
                            if (target_neighbors[t] == source) { \
                                exists = true; \
                                break; \
                            } \
                        } \
                        if (exists) continue; \
                    } \
                    target_neighbors[conn->num_neighbors[target]++] = source; \
                    source_neighbors[num_source_neighbors++] = target; \
                    hashtable[hash_idx / 32] |= (1 << (hash_idx % 32)); \
                } while (0);
                CONNECTIVITY_SEARCH(base_index + 1);
                CONNECTIVITY_SEARCH(base_index + W);
                CONNECTIVITY_SEARCH(base_index + W + 1);
                conn->num_neighbors[source] = num_source_neighbors;
            }
        }

        delete [] hashtable;
        return conn;
    }

    Connectivity* fast_slic_knn_connectivity(int H, int W, int K, const Cluster* clusters, size_t num_neighbors) {
        // auto t1 = Clock::now();
        int S = my_max((int)sqrt(H * W / K), 1);
        int nh = ceil_int(H, S), nw = ceil_int(W, S);

        std::vector< std::vector<const Cluster*> > s_cells(nh * nw);
        for (int i = 0; i < K; i++) {
            const Cluster* cluster = clusters + i;
            s_cells[(cluster->y / S) * nw + (cluster->x / S)].push_back(cluster);
        }

        // auto t2 = Clock::now();
        Connectivity* conn = new Connectivity();
        conn->num_nodes = K;
        conn->num_neighbors = new int[K];
        conn->neighbors = new uint32_t*[K];

        #pragma omp parallel for num_threads(fsparallel::nth())
        for (int i = 0; i < K; i++) {
            const Cluster* cluster = clusters + i;
            int cell_center_x = cluster->x / S, cell_center_y = cluster->y / S;

            std::vector<std::pair<int, const Cluster*>> heap;
            for (int cy = my_max(cell_center_y - 3, 0); cy < my_min(nh, cell_center_y + 3); cy++) {
                for (int cx = my_max(cell_center_x - 3, 0); cx < my_min(nw, cell_center_x + 3); cx++) {
                    for (const Cluster* cluster_around  : s_cells[cy * nw + cx])  {
                        if (cluster_around == cluster) continue;
                        int distance = fast_abs(cluster_around->x - cluster->x) + fast_abs(cluster_around->y - cluster->y);
                        if (!heap.empty() && heap.front().first <= distance) continue;

                        heap.push_back(std::pair<int, const Cluster*>(distance, cluster_around));
                        std::push_heap(heap.begin(), heap.end());
                        while (heap.size() > num_neighbors) {
                            std::pop_heap(heap.begin(), heap.end());
                            heap.pop_back();
                        }
                    }
                }
            }
            conn->num_neighbors[i] = heap.size();
            conn->neighbors[i] = new uint32_t[heap.size()];
            for (size_t j = 0; j < heap.size(); j++) {
                conn->neighbors[i][j] = heap[j].second->number;
            }
        }
        // auto t3 = Clock::now();

        // std::cerr << "Build " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "us \n";
        // std::cerr << "Find " << std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() << "us \n";
        return conn;
    }

    void fast_slic_free_connectivity(Connectivity* conn) {
        delete [] conn->num_neighbors;
        for (int i = 0; i < conn->num_nodes; i++) {
            delete [] conn->neighbors[i];
        }
        delete [] conn->neighbors;
        delete conn;
    }

    void fast_slic_get_mask_density(int H, int W, int K, const Cluster *clusters, const uint16_t* assignment, const uint8_t *mask, uint8_t *cluster_densities) {
        int *sum = new int[K];
        std::fill_n(sum, K, 0);
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                uint16_t cluster_no = assignment[W * i + j];
                if (cluster_no < (uint16_t)K)
                    sum[cluster_no] += mask[W * i + j];
            }
        }

        for (int k = 0; k < K; k++) {
            cluster_densities[k] = (uint8_t)my_min<int>(255, sum[k] / my_max(clusters[k].num_members, 1u));
        }
        delete [] sum;
    }

    void fast_slic_cluster_density_to_mask(int H, int W, int K, const Cluster *clusters, const uint16_t* assignment, const uint8_t *cluster_densities, uint8_t *result) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                uint16_t cluster_no = assignment[W * i + j];
                if (cluster_no < (uint16_t)K)
                    result[W * i + j] = cluster_densities[cluster_no];
                else
                    result[W * i + j] = 0;
            }
        }
    }
}

#ifdef PROTOTYPE_MAIN_DEMO
#include <string>
#include <chrono>
#include <fstream>
#include <memory>
#include <iostream>
typedef std::chrono::high_resolution_clock Clock;
int main(int argc, char** argv) {
    int K = 100;
    int compactness = 5;
    int max_iter = 2;
    int subsample_stride = 7;
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
    Cluster clusters[K];
    std::unique_ptr<uint8_t[]> image { new uint8_t[H * W * 3] };
    std::unique_ptr<uint16_t[]> assignment { new uint16_t[H * W] };

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

    auto t1 = Clock::now();
    fast_slic_initialize_clusters(H, W, K, image.get(), clusters);
    fast_slic_iterate(H, W, K, compactness, 0.1, subsample_stride, max_iter, image.get(), clusters, assignment.get());

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
