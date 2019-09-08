#ifndef _FAST_SLIC_RECORDER_H
#define _FAST_SLIC_RECORDER_H
#include <vector>
#include <string>
#include <cstdint>
#include <sstream>
#include <algorithm>
#include "simd-helper.hpp"

namespace fslic {
    template <class DistType>
    struct RecorderSnapshot {
        int iteration;
        std::vector<uint16_t> assignment;
        std::vector<DistType> min_dists;
        std::vector<Cluster> clusters;

        void gen(std::stringstream& stream) {
            stream << "{\"iteration\": " << iteration;
            stream << ", \"clusters\": [";

            for (int i = 0; i < (int)clusters.size(); i++) {
                Cluster &cluster = clusters[i];
                if (i > 0) stream << ",";
                stream << "{\"yx\": [" << cluster.y << "," << cluster.x << "]";
                stream << ", \"color\": [" << cluster.r << "," << cluster.g << "," << cluster.b << "]";
                stream << ", \"is_updatable\": " << (int)cluster.is_updatable;
                stream << ", \"is_active\": " << (int)cluster.is_active;
                stream << ", \"number\": " << cluster.number;
                stream << ", \"num_members\": " << cluster.num_members;
                stream << "}";
            }
            stream << "]";
            stream << ", \"assignment\": [";
            for (int i = 0; i < (int)assignment.size(); i++) {
                if (i > 0) stream << ",";
                stream << assignment[i];
            }
            stream << "]";
            stream << ", \"min_dists\": [";
            for (int i = 0; i < (int)min_dists.size(); i++) {
                if (i > 0) stream << ",";
                stream << min_dists[i];
            }
            stream << "]";
            stream << "}";
        }
    };

    template <class DistType>
    class Recorder {
    private:
        std::vector<RecorderSnapshot<DistType>> snapshots;
        int H, W, K;
        bool enabled;
    public:
        Recorder(int H, int W, int K) : H(H), W(W), K(K), enabled(false) {};
        void initialize(bool enabled) {
            this->enabled = enabled;
        }
        void push(
            int iter,
            const simd_helper::AlignedArray<uint16_t> &assignment,
            const simd_helper::AlignedArray<DistType> &min_dists,
                const Cluster *clusters) {
            if (!enabled) return;
            RecorderSnapshot<DistType> snapshot;
            snapshot.iteration = iter;
            snapshot.assignment.resize(H * W);
            snapshot.min_dists.resize(H * W);
            snapshot.clusters.resize(K);

            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    snapshot.assignment[W * i + j] = assignment.get(i, j);
                    snapshot.min_dists[W * i + j] = min_dists.get(i, j);
                }
            }
            std::copy(clusters, clusters + K, snapshot.clusters.begin());
            snapshots.push_back(snapshot);
        }

        std::string get_report() {
            std::stringstream stream;
            gen(stream);
            return stream.str();
        }

    private:
        void gen(std::stringstream& stream) {
            stream << "{\"height\": " << H;
            stream << ", \"width\": " << W;
            stream << ", \"snapshots\": [";
            for (int i = 0; i < (int)snapshots.size(); i++) {
                if (i > 0) stream << ",";
                snapshots[i].gen(stream);
            }
            stream << "]}";
        }
    };
}

#endif
