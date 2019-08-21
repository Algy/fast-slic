#include "cca.h"
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

    bool operator <(const Edge& lhs, const Edge& rhs) {
        int lhs_order_lo, lhs_order_hi, rhs_order_lo, rhs_order_hi;
        if (lhs.order_s < lhs.order_e) {
            lhs_order_lo = lhs.order_s;
            lhs_order_hi = lhs.order_e;
        } else {
            lhs_order_hi = lhs.order_s;
            lhs_order_lo = lhs.order_e;
        }
        if (rhs.order_s < rhs.order_e) {
            rhs_order_lo = rhs.order_s;
            rhs_order_hi = rhs.order_e;
        } else {
            rhs_order_hi = rhs.order_s;
            rhs_order_lo = rhs.order_e;
        }
        return lhs_order_lo < rhs_order_lo || (lhs_order_lo == rhs_order_lo && lhs_order_hi < rhs_order_hi);
    }

    bool operator ==(const Edge& lhs, const Edge& rhs) {
        int lhs_order_lo, lhs_order_hi, rhs_order_lo, rhs_order_hi;
        if (lhs.order_s < lhs.order_e) {
            lhs_order_lo = lhs.order_s;
            lhs_order_hi = lhs.order_e;
        } else {
            lhs_order_hi = lhs.order_s;
            lhs_order_lo = lhs.order_e;
        }
        if (rhs.order_s < rhs.order_e) {
            rhs_order_lo = rhs.order_s;
            rhs_order_hi = rhs.order_e;
        } else {
            rhs_order_hi = rhs.order_s;
            rhs_order_lo = rhs.order_e;
        }
        return lhs_order_lo == rhs_order_lo && lhs_order_hi == rhs_order_hi;
    }

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
            std::vector<int> component_area;
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

            std::vector<int> component_area;
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
            if (strict)
                gather_mergable_strict();
            else
                gather_mergable_not_strict();
            mark_segments_as_mergable();
            gather_adj_components();
            num_components_of_interest = (int)component_of_interest_nos.size();
            for (int order = 0; order < num_components_of_interest; order++) {
                component_no_to_order[component_of_interest_nos[order]] = order;
            }
        }

        void gather_component_features_of_intrest() {
            std::vector<std::vector<RowSegment*>> component_segs(num_components_of_interest);
            features.resize(ndims * num_components_of_interest, 0.0f);
            weights.resize(num_components_of_interest, 1.0f);

            #pragma omp parallel for
            for (int ix = 0; ix < segment_set.size(); ix++) {
                component_no_t component_no = cc_set.component_assignment[ix];
                const auto it = component_no_to_order.find(component_no);
                const auto end = component_no_to_order.end();
                if (it == end) continue;
                int order = it->second;
                auto &segs = component_segs[order];
                #pragma omp critical
                segs.push_back(&data[ix]);
            }
            #pragma omp parallel for
            for (int order = 0; order < num_components_of_interest; order++) {
                std::vector<RowSegment*>& segs = component_segs[order];
                label_no_t label = data[cc_set.component_leaders[component_of_interest_nos[order]]].label;
                kernel_fn(&segs[0], (int)segs.size(), label, &features[ndims * order], &weights[order]);
            }
        }

        void gather_edges() {
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
            std::sort(edges.begin(), edges.end());
            auto last = std::unique(edges.begin(), edges.end());
            edges.erase(last, edges.end());


            #pragma omp parallel for
            for (int i = 0; i < (int)edges.size(); i++) {
                Edge &edge = edges[i];
                edge.dist = get_distance(edge.order_s, edge.order_e);
            }
        }

        void init() {
            gather_components();
            gather_component_features_of_intrest();
            gather_edges();
        }

    public:
        /*
        struct edge_pointer_dist_less {
            bool operator()(const Edge& lhs, const Edge& rhs) {
                return lhs.dist < rhs.dist;
            }
        };

        std::vector<label_no_t> run_agglomerative_clustering() {

            DisjointSet disjoint_set(num_components_of_interest);

            // min-heap
            std::vector<Edge> curr_edge_list(edges.begin(), edges.end());
            std::vector<float> curr_features(features.begin(), features.end());
            std::vector<float> curr_weights(weights.begin(), weights.end());
            std::vector<bool> curr_mergable(is_component_mergable.begin(), is_component_mergable.end());
            std::vector<bool> already_merged(false, num_components_of_interest);

            std::vector< std::list<int> > edge_indices(num_components_of_interest);

            for (const Edge &edge : edges) {
                queue.push(edge);
                edge_indices[edge.order_s].push_back(edge.order_e);
                edge_indices[edge.order_e].push_back(edge.order_s);
            }

            while (!queue.empty()) {
                Edge edge = queue.pop();
                if (!curr_mergable[edge.order_s] && !curr_mergable[edge.order_e]) {
                    continue;
                } else if (disjoint_set.find(edge.order_s) == disjoint_set.fnid(edge.order_e)) {
                    continue;
                }

                if (!curr_mergable[edge.order_e] && curr_mergable[edge.order_e]) {
                    curr_mergable[edge.order_s] = false;
                }
                disjoint_set.merge(edge.order_s, edge.order_e);
                already_merged[edge.order_e] = true;

                for (int i = 0; i < ndims; i++) {
                    curr_features[edge.order_s] = (
                        curr_features[ndims * edge.order_s + i] * curr_weights[edge.order_s] +
                        curr_features[ndims * edge.order_e + i] * curr_weights[edge.order_e]
                    ) / (curr_weights[edge.order_s] + curr_weights[edge.order_e]);
                    curr_weights[edge.order_s] = curr_weights[edge.order_s] + curr_weights[edge.order_e];
                }

                for (int target : edge_indices[edge.order_s]) {
                }
                for (int target : edge_indices[edge.order_e]) {
                }
            }


            features[(order + 1)];
            #pragma omp parallel for
            for (int order = 0; order < num_components_of_interest; order++) {
            }
        }
        */

        std::vector<std::list<Edge>> get_edge_list() {
            std::vector<std::list<Edge>> edge_list(num_components_of_interest);
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

        std::vector<label_no_t> run_prim() {
            struct edge_cmp {
                bool operator()(const Edge& lhs, const Edge &rhs) {
                    return lhs.dist > rhs.dist;
                }
            };

            std::vector<int> root_map(num_components_of_interest);
            for (int i = 0; i < (int)root_map.size(); i++) root_map[i] = i;
            std::vector<int> visited(num_components_of_interest, 0);
            const std::vector<std::list<Edge>> edge_list = get_edge_list();
            const std::vector< std::vector<int> > component_groups = get_indepedent_component_groups();
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < (int)component_groups.size(); i++) {
                std::priority_queue<Edge, std::vector<Edge>, edge_cmp> queue;
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
        // auto t0 = Clock::now();
        DisjointSet disjoint_set;
        assign_disjoint_set(segment_set, disjoint_set);
        // auto t1 = Clock::now();
        std::unique_ptr<ComponentSet> cc_set { disjoint_set.flatten() };
        // auto t3 = Clock::now();

        AdjMerger adj_merger(segment_set, *cc_set, max_label_size, kernel_fn, min_threshold, strict);

        std::vector<label_no_t> substitute = adj_merger.run_prim();

        // auto t4 = Clock::now();


        // auto t5 = Clock::now();
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

        // auto t6 = Clock::now();

        // std::cerr << "  disjoint: " << micro(t1 -t0) << "us" << std::endl;
        // std::cerr << "  flatten: " << micro(t3 - t1) << "us" << std::endl;
        // std::cerr << "  unlabel_comp: " << micro(t4 - t3) << "us" << std::endl;
        // std::cerr << "  adj: " << micro(t5 - t4) << "us" << std::endl;
        // std::cerr << "  writeback_comp: " << micro(t6 - t5) << "us" << std::endl;
    }
};
