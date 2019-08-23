#include "cca.h"
#include <limits>
#include <algorithm>
#include <iostream>
#include <atomic>
#include <string>
#include <chrono>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <list>
#include <ctime>
#include <queue>
#include <deque>
#include <cmath>

typedef std::chrono::high_resolution_clock Clock;

template <typename T>
static int micro(T o) {
    return std::chrono::duration_cast<std::chrono::microseconds>(o).count();
}
#ifdef _OPENMP
#include <omp.h>
class OMPLock {
private:
    omp_lock_t lock;
public:
    OMPLock() { omp_init_lock(&lock); };
    void acquire() { omp_set_lock(&lock); };
    void release() { omp_unset_lock(&lock); };
    ~OMPLock() { omp_destroy_lock(&lock); };
};
#else
class OMPLock {
    void acquire() {};
    void release() {};
};
#endif

namespace cca {
    struct Edge {
        int order_s, order_e;
        float dist;
        Edge(int order_s, int order_e, float dist = 0) : order_s(order_s), order_e(order_e), dist(dist) {};
    };

    bool operator<(const Edge& lhs, const Edge& rhs) {
        int lhs_order_lo = lhs.order_s, lhs_order_hi = lhs.order_e;
        int rhs_order_lo = rhs.order_s, rhs_order_hi = rhs.order_e;
        if (lhs_order_lo > lhs_order_hi) std::swap(lhs_order_lo, lhs_order_hi);
        if (rhs_order_lo > rhs_order_hi) std::swap(rhs_order_lo, rhs_order_hi);

        return lhs_order_lo < rhs_order_lo || (lhs_order_lo == rhs_order_lo && lhs_order_hi < rhs_order_hi);
    }

    bool operator==(const Edge& lhs, const Edge& rhs) {
        int lhs_order_lo = lhs.order_s, lhs_order_hi = lhs.order_e;
        int rhs_order_lo = rhs.order_s, rhs_order_hi = rhs.order_e;
        if (lhs_order_lo > lhs_order_hi) std::swap(lhs_order_lo, lhs_order_hi);
        if (rhs_order_lo > rhs_order_hi) std::swap(rhs_order_lo, rhs_order_hi);
        return lhs_order_lo == rhs_order_lo && lhs_order_hi == rhs_order_hi;
    }

    struct edge_cmp_less {
        bool operator()(const Edge& lhs, const Edge &rhs) {
            return lhs.dist < rhs.dist || (lhs.dist == rhs.dist && lhs < rhs);
        }
    };

    struct edge_cmp_greater {
        bool operator()(const Edge& lhs, const Edge &rhs) {
            return lhs.dist > rhs.dist || (lhs.dist == rhs.dist && rhs < lhs);
        }
    };


    struct edge_hasher {
        std::size_t operator()(const Edge& k) const {
            // symmetric
            return std::hash<int>()(k.order_s) ^ std::hash<int>()(k.order_e);
        };
    };

    class AdjMerger {
    private:
        RowSegmentSet& segment_set;
        std::vector<RowSegment>& data;
        const std::vector<int> &row_offsets;
        ComponentSet &cc_set;
        int max_label_size;
        kernel_function& kernel_fn;
        int ndims;
        int min_threshold;
        bool strict;

        int num_components;
        std::vector<int> mergable_component_map; // ComponentNo -> bool
        std::vector<int> adjacent_component_map; // ComponentNo -> bool

        int num_components_of_interest;
        std::vector<component_no_t> component_of_interest_nos; // Order -> ComponentNo
        std::vector<bool> is_component_mergable; // Order -> bool
        std::unordered_map<component_no_t, int> component_no_to_order;// ComponentNo -> Order(int)

        std::vector<int> component_area;
        std::vector<float> features; // ndims * Order -> float[ndims]
        std::vector<float> weights; // Order -> float
        std::vector<Edge> edges;
    public:
        std::vector<label_no_t> merge_map; // ComponentNo -> LabelNo
        AdjMerger(RowSegmentSet& segment_set,
            ComponentSet &cc_set, int max_label_size, kernel_function &kernel_fn,
            int min_threshold, bool strict)
                : segment_set(segment_set),
                data(segment_set.get_mutable_data()),
                row_offsets(segment_set.get_row_offsets()),
                cc_set(cc_set),
                max_label_size(max_label_size),
                kernel_fn(kernel_fn),
                ndims(kernel_fn.get_ndims()),
                min_threshold(min_threshold),
                strict(strict),
                num_components(cc_set.get_num_components()),
                mergable_component_map(num_components, 0),
                adjacent_component_map(num_components, 0) {
            init();
        }

    private:
        void gather_mergable_not_strict() {
            estimate_component_area(segment_set, cc_set, component_area); // ComponentNo -> int (area)
            #pragma omp parallel for
            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                if (component_area[component_no] < min_threshold) {
                    mergable_component_map[component_no] = 1;
                    #pragma omp critical
                    {
                        component_of_interest_nos.push_back(component_no);
                        is_component_mergable.push_back(true);
                    }
                }
            }
        }

        void gather_mergable_strict() {
            struct component_area_pair {
                component_no_t component_no;
                int area;
                component_area_pair(component_no_t component_no, int area)
                    : component_no(component_no), area(area) {};
            };

            struct sort_by_area_desc {
                bool operator()(const component_area_pair& lhs, const component_area_pair &rhs) {
                    return lhs.area > rhs.area;
                }
            };

            estimate_component_area(segment_set, cc_set, component_area); // ComponentNo -> int (area)
            std::vector<component_area_pair> area_pairs;
            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                area_pairs.push_back(
                    component_area_pair(component_no, component_area[component_no])
                );
            }
            std::sort(area_pairs.begin(), area_pairs.end(), sort_by_area_desc());
            int end = (int)area_pairs.size();
            if (end > max_label_size) end = max_label_size;
            int cutoff = end;
            for (int i = 0; i < end; i++) {
                if (area_pairs[i].area < min_threshold) {
                    cutoff = i;
                    break;
                }
            }
            for (int i = cutoff; i < (int)area_pairs.size(); i++) {
                component_no_t component_no = area_pairs[i].component_no;
                component_of_interest_nos.push_back(component_no);
                is_component_mergable.push_back(true);
                mergable_component_map[component_no] = 1;
            }
        }

        void gather_adj_components() {
            int height = segment_set.get_height();
            #pragma omp parallel
            {
                #pragma omp for
                for (int i = 0; i < height; i++) {
                    int off_begin = row_offsets[i];
                    int off_end = row_offsets[i + 1];

                    for (int off = off_begin + 1; off < off_end; off++) {
                        auto &left_seg = data[off - 1];
                        auto &curr_seg = data[off];
                        if (left_seg.mergable && !curr_seg.mergable) {
                            adjacent_component_map[cc_set.component_assignment[off]] = 1;
                        } else if (curr_seg.mergable && !left_seg.mergable) {
                            adjacent_component_map[cc_set.component_assignment[off - 1]] = 1;
                        }
                    }
                }

                #pragma omp for
                for (int y = 1; y < height; y++) {
                    const int up_ix_begin = row_offsets[y - 1], up_ix_end = row_offsets[y];
                    const int curr_ix_begin = row_offsets[y], curr_ix_end = row_offsets[y + 1];
                    int up_ix = up_ix_begin, curr_ix = curr_ix_begin;
                    while (up_ix < up_ix_end && curr_ix < curr_ix_end) {
                        const RowSegment &up_seg = data[up_ix];
                        const RowSegment &curr_seg = data[curr_ix];
                        if (data[up_ix].x_end < data[curr_ix].x) {
                            up_ix++;
                        } else if (data[curr_ix].x_end < data[up_ix].x) {
                            curr_ix++;
                        } else {
                            // if control flows through here, it means prev and curr overlap
                            if (curr_seg.mergable && !up_seg.mergable) {
                                adjacent_component_map[cc_set.component_assignment[up_ix]] = 1;
                            } else if (up_seg.mergable && !curr_seg.mergable) {
                                adjacent_component_map[cc_set.component_assignment[curr_ix]] = 1;
                            }
                            if (data[curr_ix].x_end < data[up_ix].x_end) {
                                curr_ix++;
                            } else {
                                up_ix++;
                            }
                        }
                    }
                }
            }

            #pragma omp parallel for
            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                if (adjacent_component_map[component_no]) {
                    #pragma omp critical
                    {
                        component_of_interest_nos.push_back(component_no);
                        is_component_mergable.push_back(false);
                    }
                }
            }
        }

        void mark_segments_as_mergable() {
            #pragma omp parallel for
            for (int ix = 0; ix < segment_set.size(); ix++) {
                component_no_t component_no = cc_set.component_assignment[ix];
                if (mergable_component_map[component_no])
                    data[ix].mergable = true;
            }
        }


        void gather_components() {
            auto t1 = Clock::now();
            if (strict)
                gather_mergable_strict();
            else
                gather_mergable_not_strict();
            auto t2 = Clock::now();
            mark_segments_as_mergable();
            auto t3 = Clock::now();
            gather_adj_components();
            auto t4 = Clock::now();
            num_components_of_interest = (int)component_of_interest_nos.size();
            for (int order = 0; order < num_components_of_interest; order++) {
                component_no_to_order[component_of_interest_nos[order]] = order;
            }
            auto t5 = Clock::now();
            // std::cerr << "      init_comp.gather_mergable: " << micro(t2 - t1) << "us" << std::endl;
            // std::cerr << "      init_comp.mark_as_mergable: " << micro(t3 - t2) << "us" << std::endl;
            // std::cerr << "      init_comp.adj_components: " << micro(t4 - t3) << "us" << std::endl;
            // std::cerr << "      init_comp.build_map: " << micro(t5 - t4) << "us" << std::endl;
        }

        void gather_component_features_of_intrest(bool set_dist = true) {
            auto t1 = Clock::now();
            std::vector<std::vector<RowSegment*>> component_segs(num_components_of_interest);
            features.resize(ndims * num_components_of_interest, 0.0f);
            weights.resize(num_components_of_interest, 1.0f);

            auto t2 = Clock::now();

            for (int ix = 0; ix < segment_set.size(); ix++) {
                component_no_t component_no = cc_set.component_assignment[ix];
                const auto it = component_no_to_order.find(component_no);
                const auto end = component_no_to_order.end();
                if (it == end) continue;
                int order = it->second;
                auto &segs = component_segs[order];
                segs.push_back(&data[ix]);
            }

            auto t3 = Clock::now();
            #pragma omp parallel for
            for (int order = 0; order < num_components_of_interest; order++) {
                std::vector<RowSegment*>& segs = component_segs[order];
                label_no_t label = data[cc_set.component_leaders[component_of_interest_nos[order]]].label;
                kernel_fn(&segs[0], (int)segs.size(), label, &features[ndims * order], &weights[order]);
            }
            auto t4 = Clock::now();
            if (set_dist) {
                #pragma omp parallel for
                for (int i = 0; i < (int)edges.size(); i++) {
                    Edge &edge = edges[i];
                    edge.dist = get_distance(edge.order_s, edge.order_e);
                }
            }
            auto t5 = Clock::now();
            // std::cerr << "      feature.alloc: " << micro(t2 - t1) << "us" << std::endl;
            // std::cerr << "      feature.gather_segs: " << micro(t3 - t2) << "us" << std::endl;
            // std::cerr << "      feature.kernel_fn: " << micro(t4 - t3) << "us" << std::endl;
            // std::cerr << "      feature.set_dist: " << micro(t5 - t4) << "us" << std::endl;
        }

        void gather_edges() {
            auto t1 = Clock::now();
            int height = segment_set.get_height();
            #pragma omp parallel
            {
                std::vector<Edge> local_edges;
                std::unordered_set<Edge, edge_hasher> visited;

                #pragma omp for
                for (int i = 0; i < height; i++) {
                    int off_begin = row_offsets[i];
                    int off_end = row_offsets[i + 1];

                    for (int off = off_begin + 1; off < off_end; off++) {
                        auto &left_seg = data[off - 1];
                        auto &curr_seg = data[off];
                        component_no_t left_com = cc_set.component_assignment[off - 1];
                        component_no_t curr_com = cc_set.component_assignment[off];
                        if (left_seg.mergable || curr_seg.mergable) {
                            int left_order, curr_order;
                            {
                                const auto it = component_no_to_order.find(left_com);
                                if (it == component_no_to_order.end()) continue;
                                left_order = it->second;
                            }
                            {
                                const auto it = component_no_to_order.find(curr_com);
                                if (it == component_no_to_order.end()) continue;
                                curr_order = it->second;
                            }
                            if (left_order != curr_order) {
                                Edge edge(left_order, curr_order);
                                if (visited.find(edge) == visited.end()) {
                                    local_edges.push_back(edge);
                                    visited.insert(edge);
                                }
                            }
                        }
                    }
                }

                #pragma omp for
                for (int y = 1; y < height; y++) {
                    const int up_ix_begin = row_offsets[y - 1], up_ix_end = row_offsets[y];
                    const int curr_ix_begin = row_offsets[y], curr_ix_end = row_offsets[y + 1];
                    int up_ix = up_ix_begin, curr_ix = curr_ix_begin;
                    while (up_ix < up_ix_end && curr_ix < curr_ix_end) {
                        const RowSegment &up_seg = data[up_ix];
                        const RowSegment &curr_seg = data[curr_ix];
                        component_no_t up_com = cc_set.component_assignment[up_ix];
                        component_no_t curr_com = cc_set.component_assignment[curr_ix];
                        if (data[up_ix].x_end < data[curr_ix].x) {
                            up_ix++;
                        } else if (data[curr_ix].x_end < data[up_ix].x) {
                            curr_ix++;
                        } else {
                            // if control flows through here, it means prev and curr overlap
                            do {
                                if (curr_seg.mergable || up_seg.mergable) {
                                    int up_order, curr_order;
                                    {
                                        const auto it = component_no_to_order.find(up_com);
                                        if (it == component_no_to_order.end()) break;
                                        up_order = it->second;
                                    }
                                    {
                                        const auto it = component_no_to_order.find(curr_com);
                                        if (it == component_no_to_order.end()) break;
                                        curr_order = it->second;
                                    }
                                    if (up_order != curr_order) {
                                        Edge edge(up_order, curr_order);
                                        if (visited.find(edge) == visited.end()) {
                                            local_edges.push_back(edge);
                                            visited.insert(edge);
                                        }
                                    }
                                }
                            } while (false);

                            if (data[curr_ix].x_end < data[up_ix].x_end) {
                                curr_ix++;
                            } else {
                                up_ix++;
                            }
                        }
                    }
                }
                #pragma omp critical
                edges.insert(edges.end(), local_edges.begin(), local_edges.end());
            }
            auto t2 = Clock::now();
            std::sort(edges.begin(), edges.end());
            auto last = std::unique(edges.begin(), edges.end());
            edges.erase(last, edges.end());
            auto t3 = Clock::now();

            // std::cerr << "      gather_edges.adj_test: " << micro(t2 - t1) << "us" << std::endl;
            // std::cerr << "      gather_edges.sort: " << micro(t3 - t2) << "us" << std::endl;
        }

        void init() {
            gather_components();
            gather_edges();
        }

    public:
        std::vector<std::vector<Edge>> get_edge_list() {
            std::vector<std::vector<Edge>> edge_list(num_components_of_interest);
            for (const Edge &edge : edges) {
                edge_list[edge.order_s].push_back(Edge(edge.order_s, edge.order_e, edge.dist));
                edge_list[edge.order_e].push_back(Edge(edge.order_e, edge.order_s, edge.dist));
            }
            return edge_list;
        }

        std::vector<std::vector<int>> get_indepedent_component_groups() {
            DisjointSet disjoint_set(num_components_of_interest);
            for (const Edge &edge : edges) {
                disjoint_set.merge(edge.order_s, edge.order_e);
            }
            std::unique_ptr<ComponentSet> component_complex { disjoint_set.flatten() };
            std::vector< std::vector<int> > component_groups(component_complex->num_components);
            for (int order = 0; order < num_components_of_interest; order++) {
                component_groups[component_complex->component_assignment[order]].push_back(order);
            }
            return component_groups;
        }

        std::vector<label_no_t> run_simple() {
            DisjointSet mergable_disjoint_set(num_components);
            std::vector<label_no_t> result(num_components, 0xFFFF);

            // relabeling
            label_no_t last_new_label = 0;
            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                if (!mergable_component_map[component_no]) {
                    label_no_t new_label = last_new_label++;
                    result[component_no] = new_label;
                }
            }
            for (const Edge &edge : edges) {
                if (is_component_mergable[edge.order_s] && is_component_mergable[edge.order_e]) {
                    mergable_disjoint_set.merge(
                        component_of_interest_nos[edge.order_s],
                        component_of_interest_nos[edge.order_e]
                    );
                }
            }
            for (const Edge &edge : edges) {
                if (!is_component_mergable[edge.order_s]) {
                    result[mergable_disjoint_set.find(component_of_interest_nos[edge.order_e])] =
                        result[component_of_interest_nos[edge.order_s]];
                } else if (!is_component_mergable[edge.order_e]) {
                    result[mergable_disjoint_set.find(component_of_interest_nos[edge.order_s])] =
                        result[component_of_interest_nos[edge.order_e]];
                }
            }
            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                if (mergable_component_map[component_no]) {
                    result[component_no] = result[mergable_disjoint_set.find(component_no)];
                }
            }
            return result;
        }

        std::vector<label_no_t> run_greedy_agglomerative_clustering() {
            auto t0 = Clock::now();
            gather_component_features_of_intrest(false);
            auto t1 = Clock::now();
            std::vector<float> curr_features(features);
            std::vector<float> curr_weights(weights);
            std::vector<bool> curr_mergable(is_component_mergable);
            std::vector<bool> removed(num_components_of_interest, false);
            std::vector<int> recovered_list;
            std::vector<int> curr_component_area(num_components_of_interest);

            for (int i = 0; i < num_components_of_interest; i++) {
                curr_component_area[i] = component_area[component_of_interest_nos[i]];
            }

            DisjointSet disjoint_set(num_components_of_interest);

            auto t20 = Clock::now();
            // const auto component_groups = get_indepedent_component_groups();
            auto t21 = Clock::now();
            auto edge_list = get_edge_list();
            auto t2 = Clock::now();

            int num_occupied = 0;
            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                if (!mergable_component_map[component_no]) {
                    num_occupied++;
                }
            }
            // std::atomic<int> superpixel_recover_capacity;
            int superpixel_recover_capacity;
            if (strict) {
                superpixel_recover_capacity = max_label_size - num_occupied;
            } else {
                superpixel_recover_capacity = std::numeric_limits<int>::max();
            }
            auto t3 = Clock::now();

            for (int i = 0; i < 1; i++) { // (int)component_groups.size(); i++) {
                std::deque<int> q;
                // const std::vector<int>& vertices = component_groups[i];
                // for (int vertex : vertices) {
                for (int vertex =  0; vertex < num_components_of_interest; vertex++) {
                    if (curr_mergable[vertex]) {
                        q.push_back(vertex);
                    }
                }

                // std::cerr << "q.size(): " << q.size() << "\n";
                // std::cerr << "num_occupied: " << num_occupied << "\n";
                while (!q.empty()) {
                    int curr_v = q.front();
                    q.pop_front();
                    // std::cerr << "q.size(): " << q.size() << "\n";
                    if (removed[curr_v]) continue;

                    float min_dist = std::numeric_limits<float>::max();
                    int min_dist_v = -1;
                    for (const Edge &target_edge : edge_list[curr_v]) {
                        int target = target_edge.order_e;
                        float new_dist = 0;
                        for (int i = 0; i < ndims; i++) {
                            float delta = (
                                curr_features[ndims * curr_v + i] -
                                curr_features[ndims * target + i]
                            );
                            new_dist += delta * delta;
                        }
                        if (min_dist > new_dist) {
                            min_dist = new_dist;
                            min_dist_v = target;
                        }
                    }
                    if (min_dist_v == -1) {
                        std::cerr << "== Invariance assertion(min_dist_v != -1) failed ==" << curr_v << "\n";
                        std::cerr << "curr_v = " << curr_v << "\n";
                        std::cerr << "q.size() = " << q.size() << "\n";
                        std::cerr << "edge_list[curr_v].size() = " << edge_list[curr_v].size() << "\n";
                        std::cerr << "min_dist = " << min_dist << "\n";
                        std::cerr << "num_component_complex = " << disjoint_set.flatten()->num_components << "\n";
                        abort();
                    }

                    int target_v = min_dist_v;
                    int next_v = curr_v;

                    if (component_area[curr_v] >= min_threshold &&
                            !curr_mergable[target_v] &&
                            superpixel_recover_capacity-- > 0) {
                        curr_mergable[curr_v] = false;
                        recovered_list.push_back(curr_v);
                        continue;
                    }

                    disjoint_set.merge(curr_v, target_v);
                    removed[target_v] = true;

                    Edge edge = Edge(curr_v, target_v);
                    auto it_l = std::find(
                        edge_list[curr_v].begin(),
                        edge_list[curr_v].end(),
                        edge
                    );
                    auto it_r = std::find(
                        edge_list[target_v].begin(),
                        edge_list[target_v].end(),
                        edge
                    );
                    edge_list[curr_v].erase(it_l);
                    edge_list[target_v].erase(it_r);

                    std::unordered_set<int> s_targets;
                    for (const Edge &neighbor : edge_list[edge.order_s]) {
                        s_targets.insert(neighbor.order_e);
                    }

                    for (const Edge &neighbor : edge_list[edge.order_e]) {
                        auto it = std::find(
                            edge_list[neighbor.order_e].begin(),
                            edge_list[neighbor.order_e].end(),
                            neighbor
                        );

                        if (s_targets.find(neighbor.order_e) == s_targets.end()) {
                            it->order_e = edge.order_s;
                            edge_list[edge.order_s].push_back(Edge(edge.order_s, neighbor.order_e));
                        } else {
                            edge_list[neighbor.order_e].erase(it);
                        }
                    }
                    edge_list[edge.order_e].clear();

                    float w_s = curr_weights[curr_v], w_e = curr_weights[target_v];
                    float sum_w = w_s + w_e;
                    float reciprocal_sum_w = 1 / sum_w;
                    for (int i = 0; i < ndims; i++) {
                        curr_features[ndims * next_v + i] = (
                            curr_features[ndims * curr_v + i] * w_s +
                            curr_features[ndims * target_v + i] * w_e
                        ) * reciprocal_sum_w;
                        /*
                        if (std::isnan(sum_w) || std::isnan(curr_features[ndims * next_v + i])) {
                            std::cerr << "nan found on " << curr_v << " <=> " << target_v << "\n";
                            std::cerr << "w_s " << w_s << "\n";
                            std::cerr << "w_e " << w_e << "\n";
                            std::cerr << "reciprocal_sum_w " << reciprocal_sum_w << "\n";
                            std::cerr << "curr_features[ndims * curr_v + i] " << curr_features[ndims * curr_v + i] << "\n";
                            std::cerr << "curr_features[ndims * target_v + i] " << curr_features[ndims * target_v + i] << "\n";
                            std::cerr << "curr_features[ndims * next_v + i] " << curr_features[ndims * next_v + i] << "\n";
                            abort();
                        }
                        */
                    }

                    int next_area = curr_component_area[curr_v] + curr_component_area[target_v];
                    bool next_mergable = curr_mergable[target_v];
                    curr_weights[next_v] = sum_w;
                    curr_component_area[next_v] = next_area;
                    curr_mergable[next_v] = next_mergable;
                    if (next_mergable) {
                        q.push_back(next_v);
                    }
                }
            }
            auto t4 = Clock::now();
            // std::cerr << "num_component_complex: " << disjoint_set.flatten()->num_components << "\n";
            std::vector<label_no_t> result(num_components, 0xFFFF);

            label_no_t last_new_label = 0;
            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                if (!mergable_component_map[component_no]) {
                    result[component_no] = last_new_label++;
                }
            }
            for (int order : recovered_list) {
                component_no_t root_component_no = component_of_interest_nos[disjoint_set.find(order)];
                result[root_component_no] = last_new_label++;
            }

            if (strict) {
                if (last_new_label > max_label_size) {
                    std::cerr << "== Invariance assertion(last_new_label > max_label_size) failed ==\n" ;
                    abort();
                }
            }

            for (int order = 0; order < num_components_of_interest; order++) {
                if (!is_component_mergable[order]) {
                    component_no_t component_no = component_of_interest_nos[order];
                    component_no_t root_component_no = component_of_interest_nos[disjoint_set.find(order)];
                    result[root_component_no] = result[component_no];
                }
            }
            for (int order = 0; order < num_components_of_interest; order++) {
                if (is_component_mergable[order]) {
                    component_no_t component_no = component_of_interest_nos[order];
                    component_no_t root_component_no = component_of_interest_nos[disjoint_set.find(order)];
                    result[component_no] = result[root_component_no];
                }
            }
            auto t5 = Clock::now();

            // std::cerr << "    features: " << micro(t1 - t0) << "us" << std::endl;
            // std::cerr << "    get_comp_groups: " << micro(t21 - t20) << "us" << std::endl;
            // std::cerr << "    get_edges: " << micro(t2 - t21) << "us" << std::endl;
            // std::cerr << "    queueing: " << micro(t4 - t3) << "us" << std::endl;
            // std::cerr << "    writeback_comp: " << micro(t5 - t4) << "us" << std::endl;
            return result;
        }

        std::vector<label_no_t> run_agglomerative_clustering() {
            gather_component_features_of_intrest();

            std::vector<float> curr_features(features.begin(), features.end());
            std::vector<float> curr_weights(weights.begin(), weights.end());
            std::vector<bool> curr_mergable(is_component_mergable.begin(), is_component_mergable.end());
            DisjointSet disjoint_set(num_components);

            const auto orig_edge_list = get_edge_list();
            auto edge_list = orig_edge_list;
            const auto component_groups = get_indepedent_component_groups();
            for (int i = 0; i < (int)component_groups.size(); i++) {
                const std::vector<int>& vertices = component_groups[i];
                // sort edges by distance in ascending order
                std::set<Edge, edge_cmp_less> queue;
                for (int order : vertices) {
                    const auto &adjs = orig_edge_list[order];
                    queue.insert(adjs.begin(), adjs.end());
                }
                while (!queue.empty()) {
                    auto it = queue.begin();
                    const Edge edge = *it;

                    queue.erase(it);
                    if (!curr_mergable[edge.order_s] && !curr_mergable[edge.order_e]) continue;

                    // order_s <-(merge) order_e
                    disjoint_set.merge(
                        component_of_interest_nos[edge.order_s],
                        component_of_interest_nos[edge.order_e]
                    );
                    curr_mergable[edge.order_s] = curr_mergable[edge.order_s] && curr_mergable[edge.order_e];

                    float w_s = curr_weights[edge.order_s], w_e = curr_weights[edge.order_e];
                    float sum_w = w_s + w_e;
                    for (int i = 0; i < ndims; i++) {
                        curr_features[ndims * edge.order_s + i] = (
                            curr_features[ndims * edge.order_s + i] * w_s +
                            curr_features[ndims * edge.order_e + i] * w_e
                        ) / sum_w;
                    }
                    curr_weights[edge.order_s] = sum_w;

                    auto it_l = std::find(
                        edge_list[edge.order_s].begin(),
                        edge_list[edge.order_s].end(),
                        edge
                    );
                    auto it_r = std::find(
                        edge_list[edge.order_e].begin(),
                        edge_list[edge.order_e].end(),
                        edge
                    );
                    edge_list[edge.order_s].erase(it_l);
                    edge_list[edge.order_e].erase(it_r);

                    std::unordered_set<int> s_targets;
                    for (Edge &neighbor : edge_list[edge.order_s]) {
                        queue.erase(neighbor);
                        s_targets.insert(neighbor.order_e);

                        float new_dist = 0;
                        for (int i = 0; i < ndims; i++) {
                            float delta = (
                                curr_features[ndims * edge.order_s + i] -
                                curr_features[ndims * neighbor.order_e + i]
                            );
                            new_dist += delta * delta;
                        }
                        auto it = std::find(
                            edge_list[neighbor.order_e].begin(),
                            edge_list[neighbor.order_e].end(),
                            neighbor
                        );
                        it->dist = new_dist;
                        neighbor.dist = new_dist;
                        queue.insert(neighbor);
                    }

                    for (Edge neighbor : edge_list[edge.order_e]) {
                        queue.erase(neighbor);
                        auto it = std::find(
                            edge_list[neighbor.order_e].begin(),
                            edge_list[neighbor.order_e].end(),
                            neighbor
                        );

                        if (s_targets.find(neighbor.order_e) == s_targets.end()) {
                            float new_dist = 0;
                            for (int i = 0; i < ndims; i++) {
                                float delta = (
                                    curr_features[ndims * edge.order_s + i] -
                                    curr_features[ndims * neighbor.order_e + i]
                                );
                                new_dist += delta * delta;
                            }
                            it->order_e = edge.order_s;
                            it->dist = new_dist;

                            neighbor.order_s = edge.order_s;
                            neighbor.dist = new_dist;
                            queue.insert(neighbor);
                            edge_list[edge.order_s].push_back(neighbor);
                        } else {
                            edge_list[neighbor.order_e].erase(it);
                        }
                    }
                    edge_list[edge.order_e].clear();
                }
            }

            std::vector<label_no_t> result(num_components, 0xFFFF);
            // relabeling
            label_no_t last_new_label = 0;
            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                if (!mergable_component_map[component_no]) {
                    label_no_t new_label = last_new_label++;
                    result[component_no] = new_label;
                    if (adjacent_component_map[component_no]) {
                        result[disjoint_set.find(component_no)] = new_label;
                    }
                }
            }

            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                if (mergable_component_map[component_no]) {
                    result[component_no] = result[disjoint_set.find(component_no)];
                }
            }
            return result;
        }

        std::vector<label_no_t> run_prim() {
            gather_component_features_of_intrest();

            std::vector<int> root_map(num_components_of_interest);
            for (int i = 0; i < (int)root_map.size(); i++) root_map[i] = i;
            std::vector<int> visited(num_components_of_interest, 0);
            const std::vector<std::vector<Edge>> edge_list = get_edge_list();
            const std::vector< std::vector<int> > component_groups = get_indepedent_component_groups();
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < (int)component_groups.size(); i++) {
                std::priority_queue<Edge, std::vector<Edge>, edge_cmp_greater> queue;
                const std::vector<int>& vertices = component_groups[i];

                for (int order : vertices) {
                    if (is_component_mergable[order]) continue;
                    visited[order] = 1;

                    for (Edge edge : edge_list[order]) {
                        queue.push(edge);
                    }
                }

                while (!queue.empty()) {
                    const Edge edge = queue.top();
                    queue.pop();
                    if (visited[edge.order_e]) continue;
                    visited[edge.order_e] = 1;
                    root_map[edge.order_e] = root_map[edge.order_s];
                    // Relax
                    for (const Edge & neighbor: edge_list[edge.order_e]) {
                        if (visited[neighbor.order_e]) continue;
                        queue.push(neighbor);
                    }
                }
            }
            std::vector<label_no_t> result(num_components, 0xFFFF);
            // relabeling
            label_no_t last_new_label = 0;
            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                if (!mergable_component_map[component_no]) {
                    result[component_no] = last_new_label++;
                }
            }
            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                if (mergable_component_map[component_no]) {
                    result[component_no] =
                        result[component_of_interest_nos[root_map[component_no_to_order[component_no]]]];
                }
            }
            return result;
        }
    private:
        float get_distance(int lhs_order, int rhs_order) {
            const float *lhs_feats = &features[ndims * lhs_order];
            const float *rhs_feats = &features[ndims * rhs_order];

            float dist = 0;
            for (int i = 0; i < ndims; i++) {
                dist += (lhs_feats[i] - rhs_feats[i]) * (lhs_feats[i] - rhs_feats[i]);
            }
            return dist;
        }
    };

    RowSegmentSet& RowSegmentSet::operator+=(const RowSegmentSet &rhs) {
        int offset_start = row_offsets.back(); row_offsets.pop_back();
        std::transform(
            rhs.row_offsets.begin(),
            rhs.row_offsets.end(),
            std::back_inserter(row_offsets),
            [offset_start](int x) { return offset_start + x; }
        );
        std::copy(rhs.data.begin(), rhs.data.end(), std::back_inserter(data));
        return *this;
    }

    void RowSegmentSet::set_from_2d_array(const label_no_t *labels, int H, int W) {
        width = W;
        int num_segments = 0;
        row_offsets.resize(H + 1);

        #pragma omp parallel
        {
            int first_row_idx = -1;
            std::vector<RowSegment> local_data;

            #pragma omp for schedule(static)
            for (int i = 0; i < H; i++) {
                if (first_row_idx == -1) first_row_idx = i;

                int num_segs_in_row = 0;
                for (int j = 0; j < W;) {
                    label_no_t label_hd = labels[W * i + j];
                    int j_start = j++;
                    while (j < W && labels[W * i + j] == label_hd) j++;
                    local_data.push_back(RowSegment(label_hd, i, j_start, j));
                    num_segs_in_row++;
                }
                row_offsets[i + 1] = num_segs_in_row;
            }

            #pragma omp atomic
            num_segments += local_data.size();

            #pragma omp barrier
            #pragma omp single
            {
                data.resize(num_segments);
                row_offsets[0] = 0;
                for (int i = 0; i < H; i++) row_offsets[i + 1] += row_offsets[i];
            }

            if (first_row_idx != -1)
                std::copy(local_data.begin(), local_data.end(), data.begin() + row_offsets[first_row_idx]);
        }
    }

    void RowSegmentSet::collapse() {
        int H = get_height();
        #pragma omp parallel for
        for (int i = 0; i < H; i++) {
            int off_end = row_offsets[i + 1];
            for (int off = row_offsets[i]; off < off_end;) {
                label_no_t label_hd = data[off].label;
                int off_st = off++;
                while (off < off_end && data[off].label == label_hd) off++;

                if (off_st == off - 1) continue;

                auto last_x_end = data[off - 1].x_end;
                data[off_st].x_end = last_x_end;
                for (int t = off_st + 1; t < off; t++) data[t].x = data[t].x_end = last_x_end;
            }
        }
    }

    void merge_segment_rows(const RowSegmentSet &segment_set, DisjointSet &disjoint_set, int y) {
        const auto &row_offsets = segment_set.get_row_offsets();
        const auto &data = segment_set.get_data();

        const int up_ix_begin = row_offsets[y - 1], up_ix_end = row_offsets[y];
        const int curr_ix_begin = row_offsets[y], curr_ix_end = row_offsets[y + 1];
        int up_ix = up_ix_begin, curr_ix = curr_ix_begin;

        while (up_ix < up_ix_end && curr_ix < curr_ix_end) {
            if (data[up_ix].x_end <= data[curr_ix].x) {
                up_ix++;
            } else if (data[curr_ix].x_end <= data[up_ix].x) {
                curr_ix++;
            } else {
                // if control flows through here, it means up and curr segment overlap
                if (data[up_ix].label == data[curr_ix].label) {
                    disjoint_set.merge(up_ix, curr_ix);
                }
                if (data[curr_ix].x_end < data[up_ix].x_end) {
                    curr_ix++;
                } else {
                    up_ix++;
                }
            }
        }
    }

    void estimate_component_area(const RowSegmentSet &segment_set, const ComponentSet &cc_set, std::vector<int> &dest) {
        int num_components = cc_set.get_num_components();
        dest.resize(num_components, 0); // dest: ComponentNo -> int (area)
        const auto &data = segment_set.get_data();

        #pragma omp parallel
        {
            std::vector<int> local_component_area(num_components, 0);
            #pragma omp for
            for (segment_no_t seg_no = 0; seg_no < segment_set.size(); seg_no++) {
                component_no_t component_no = cc_set.component_assignment[seg_no];
                local_component_area[component_no] += data[seg_no].get_length();
            }

            #pragma omp critical
            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                dest[component_no] += local_component_area[component_no];
            }
        }
    }

    void assign_disjoint_set(const RowSegmentSet &segment_set, DisjointSet &dest) {
        while (dest.size < segment_set.size()) dest.add();
        std::vector<int> seam_ys;
        int height = segment_set.get_height();
        #pragma omp parallel
        {
            int seam = -1;
            #pragma omp for schedule(static)
            for (int i = 0; i < height; i++) {
                if (seam == -1) {
                    seam = i;
                    continue;
                }
                merge_segment_rows(segment_set, dest, i);
            }

            #pragma omp critical
            seam_ys.push_back(seam);
        }
        for (int i : seam_ys) {
            if (i <= 0) continue;
            merge_segment_rows(segment_set, dest, i);
        }
    }

    std::unique_ptr<ComponentSet> DisjointSet::flatten() {
        int size = (int)parents.size();
        std::unique_ptr<ComponentSet> result { new ComponentSet(size) };
        std::atomic<int> component_counter { 0 };

        #pragma omp parallel
        {
            // First, rename leading nodes
            #pragma omp for
            for (tree_node_t i = 0; i < size; i++) {
                if (parents[i] == i) {
                    result->component_assignment[i] = component_counter++;
                }
            }

            // Second, allocate info arrays
            #pragma omp single
            {
                result->num_components = component_counter.load();
                result->num_component_members.resize(result->num_components, 0);
                result->component_leaders.resize(result->num_components);
            }

            std::vector<int> local_num_component_members;
            local_num_component_members.resize(result->num_components, 0);

            #pragma omp for
            for (tree_node_t i = 0; i < size; i++) {
                tree_node_t parent = parents[i];
                if (parent < i) {
                    component_no_t component_no = result->component_assignment[parent];
                    // In case that parent crosses over thread boundaries, it could be possible
                    // that component_no is not assigned. If so, search for the value of it walking through tree upward.
                    while (component_no == -1) {
                        parent = parents[parent];
                        component_no = result->component_assignment[parent];
                    }
                    result->component_assignment[i] = component_no;
                    local_num_component_members[component_no]++;
                } else {
                    component_no_t component_no = result->component_assignment[i];
                    result->component_leaders[component_no] = i;
                    local_num_component_members[component_no]++;
                }
            }

            #pragma omp critical
            for (component_no_t i = 0; i < result->num_components; i++) {
                result->num_component_members[i] += local_num_component_members[i];
            }
        }
        return std::move(result);
    }

    ConnectivityEnforcer::ConnectivityEnforcer(const uint16_t *labels, int H,
        int W, int K, int min_threshold,
        kernel_function& kernel_fn, bool strict)
            : min_threshold(min_threshold), max_label_size(K), strict(strict), kernel_fn(kernel_fn) {
        // auto t0 = Clock::now();
        segment_set.set_from_2d_array(labels, H, W);
        // auto t1 = Clock::now();
        // std::cerr << "  row segmentation: " << micro(t1 -t0) << "us" << std::endl;
    }

    class default_kernel_fn_cls : public kernel_function {
    public:
        virtual int get_ndims() { return 2; };
        virtual void operator() (RowSegment** segments, int size, label_no_t label, float* out, float* out_weight) {
            int ndims = get_ndims();
            float* __restrict my_out = out;
            std::fill_n(my_out, ndims, 0.0f);

            int counter = 0;
            for (int i = 0; i < size; i++) {
                const cca::RowSegment& seg = *segments[i];
                for (int x = seg.x; x < seg.x_end; x++) {
                    my_out[0] += seg.y;
                    my_out[1] += seg.x;
                }
                counter += seg.get_length();
            }
            for (int i = 0; i < ndims; i++) my_out[i] /= counter;
            *out_weight = (float)counter;
        }
    } default_kernel_fn;

    ConnectivityEnforcer::ConnectivityEnforcer(const uint16_t *labels, int H,
        int W, int K, int min_threshold, bool strict) :
            ConnectivityEnforcer(
                labels, H, W, K, min_threshold,
                default_kernel_fn,
                strict) {};

    void ConnectivityEnforcer::execute(label_no_t *out) {
        auto t0 = Clock::now();
        DisjointSet disjoint_set;
        assign_disjoint_set(segment_set, disjoint_set);
        auto t1 = Clock::now();
        std::unique_ptr<ComponentSet> cc_set { disjoint_set.flatten() };
        auto t2 = Clock::now();

        AdjMerger adj_merger(segment_set, *cc_set, max_label_size, kernel_fn, min_threshold, strict);
        auto t3 = Clock::now();
        std::vector<label_no_t> substitute = adj_merger.run_greedy_agglomerative_clustering();// adj_merger.run_simple(); // adj_merger.run_agglomerative_clustering();// adj_merger.run_simple();

        auto t4 = Clock::now();


        int width = segment_set.get_width(), height = segment_set.get_height();
        const auto &row_offsets = segment_set.get_row_offsets();
        const auto &data = segment_set.get_data();
        #pragma omp parallel for
        for (int i = 0; i < height; i++) {
            for (int off = row_offsets[i]; off < row_offsets[i + 1]; off++) {
                const RowSegment &segment = data[off];
                component_no_t component_no = cc_set->component_assignment[off];
                label_no_t label_subs = substitute[component_no];
                if (label_subs == 0xFFFF) label_subs = 0; // fallback
                for (int j = segment.x; j < segment.x_end; j++) {
                    out[i * width + j] = label_subs;
                }
            }
        }
        auto t5 = Clock::now();

        // std::cerr << "  disjoint: " << micro(t1 -t0) << "us" << std::endl;
        // std::cerr << "  flatten: " << micro(t2 - t1) << "us" << std::endl;
        // std::cerr << "  init comp: " << micro(t3 - t2) << "us" << std::endl;
        // std::cerr << "  aggclustering: " << micro(t4 - t3) << "us" << std::endl;
        // std::cerr << "  writeback_comp: " << micro(t5 - t4) << "us" << std::endl;
    }
};
