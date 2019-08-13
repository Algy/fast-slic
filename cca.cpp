#include "cca.h"
#include <algorithm>
#include <iostream>
#include <atomic>
#include <string>
#include <chrono>
#include <set>
#include <limits>
#include <ctime>

typedef std::chrono::high_resolution_clock Clock;

template <typename T>
static int micro(T o) {
    return std::chrono::duration_cast<std::chrono::microseconds>(o).count();
}

namespace cca {
    class AdjMerger {
    private:
        const RowSegmentSet &segment_set;
        const ComponentSet &cc_set;
        std::vector<int> largest_adj_area; // ComponentNo -> area
        std::vector<label_no_t> largest_adj_label; // ComponentNo -> Label
        std::vector<int> component_area;
        std::vector< std::set<component_no_t> > undef_neighbors; // ComponentNo -> std::set<ComponentNo>
    public:
        AdjMerger(const RowSegmentSet &segment_set, const ComponentSet &cc_set) :
                segment_set(segment_set), cc_set(cc_set),
                largest_adj_area(cc_set.get_num_components(), 0),
                largest_adj_label(cc_set.get_num_components(), 0xFFFF),
                undef_neighbors(cc_set.get_num_components()) {
            estimate_component_area(segment_set, cc_set, component_area); // ComponentNo -> int (area)
        };
        void update(segment_no_t target_ix, segment_no_t adj_ix, label_no_t label) {
            component_no_t target = cc_set.component_assignment[target_ix];
            component_no_t adj = cc_set.component_assignment[adj_ix];
            if (largest_adj_area[target] < component_area[adj]) {
                largest_adj_area[target] = component_area[adj];
                largest_adj_label[target] = label;
            }
        };

        void add_undef_neighbor(segment_no_t target_ix, segment_no_t adj_ix) {
            component_no_t target = cc_set.component_assignment[target_ix];
            component_no_t adj = cc_set.component_assignment[adj_ix];
            if (target != adj) {
                undef_neighbors[target].insert(adj);
                undef_neighbors[adj].insert(target);
            }
        }
        void concat(const AdjMerger &rhs) {
            #pragma omp parallel
            {
                #pragma omp for
                for (int i = 0; i < (int)largest_adj_area.size(); i++) {
                    if (largest_adj_area[i] < rhs.largest_adj_area[i]) {
                        largest_adj_area[i] = rhs.largest_adj_area[i];
                        largest_adj_label[i] = rhs.largest_adj_label[i];
                    }
                }
                #pragma omp for
                for (int i = 0; i < (int)undef_neighbors.size(); i++) {
                    undef_neighbors[i].insert(rhs.undef_neighbors[i].begin(), rhs.undef_neighbors[i].end());
                }
            }
        };

        void group_neighbors() {
            int num_components = cc_set.get_num_components();
            std::vector< std::vector<component_no_t> > groups;

            #pragma omp parallel
            {
                std::vector<component_no_t> stack;
                std::vector<bool> visited(num_components, false);

                #pragma omp for
                for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                    if (undef_neighbors[component_no].empty()) continue;
                    if (visited[component_no]) continue;
                    stack.push_back(component_no);

                    std::vector<component_no_t> group;
                    while (!stack.empty()) {
                        component_no_t curr_component = stack.back();
                        stack.pop_back();
                        if (visited[curr_component]) continue;
                        visited[curr_component] = true;
                        group.push_back(curr_component);
                        for (component_no_t neighbor_no : undef_neighbors[curr_component]) {
                            stack.push_back(neighbor_no);
                        }
                    }

                    #pragma omp critical
                    groups.push_back(std::move(group));
                }
            }
            for (int i = 0; i < (int)groups.size(); i++) {
                const std::vector<component_no_t> &group = groups[i];
                int max_area = std::numeric_limits<int>::min();
                label_no_t max_label = 0xFFFF;
                for (component_no_t c : group) {
                    if (max_area < largest_adj_area[c]) {
                        max_area = largest_adj_area[c];
                        max_label = largest_adj_label[c];
                    }
                }
                for (component_no_t c : group) {
                    largest_adj_area[c] = max_area;
                    largest_adj_label[c] = max_label;
                }
            }
        }


        void copy_back(std::vector<label_no_t> &dest) {
            dest = largest_adj_label;
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
                    local_data.push_back(RowSegment(label_hd, j_start, j));
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

    template <bool seam_concat>
    void merge_segment_rows(const RowSegmentSet &segment_set, DisjointSet &disjoint_set, int y) {
        const auto &row_offsets = segment_set.get_row_offsets();
        const auto &data = segment_set.get_data();

        const int up_ix_begin = row_offsets[y - 1], up_ix_end = row_offsets[y];
        const int curr_ix_begin = row_offsets[y], curr_ix_end = row_offsets[y + 1];
        int up_ix = up_ix_begin, curr_ix = curr_ix_begin;
        bool curr_first_occurrence = true;

        while (up_ix < up_ix_end && curr_ix < curr_ix_end) {
            if (data[up_ix].x_end <= data[curr_ix].x) {
                up_ix++;
            } else if (data[curr_ix].x_end <= data[up_ix].x) {
                curr_ix++;
                curr_first_occurrence = true;
            } else {
                // if control flows through here, it means up and curr segment overlap
                if (data[up_ix].label == data[curr_ix].label) {
                    if (seam_concat) {
                        disjoint_set.merge(up_ix, curr_ix);
                    } else {
                        if (curr_first_occurrence) {
                            disjoint_set.add_single(up_ix, curr_ix);
                            curr_first_occurrence = false;
                        } else {
                            disjoint_set.merge(up_ix, curr_ix);
                        }
                    }
                }
                if (data[curr_ix].x_end < data[up_ix].x_end) {
                    curr_ix++;
                    curr_first_occurrence = true;
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


    void unlabeled_adj(const RowSegmentSet &segment_set, const ComponentSet &cc_set, std::vector<label_no_t> &dest) {
        auto t01 = Clock::now();
        int num_components = cc_set.get_num_components();
        const auto &row_offsets = segment_set.get_row_offsets();
        const auto &data = segment_set.get_data();
        auto height = segment_set.get_height();

        std::vector< std::unique_ptr<AdjMerger> > local_mergers;
        // left to right
        auto t0 = Clock::now();
        #pragma omp parallel
        {
            AdjMerger *local_merger = new AdjMerger(segment_set, cc_set);
            #pragma omp critical
            local_mergers.push_back(std::unique_ptr<AdjMerger>(local_merger));
            #pragma omp barrier

            #pragma omp for
            for (int i = 0; i < height; i++) {
                int off_begin = row_offsets[i];
                int off_end = row_offsets[i + 1];

                for (int off = off_begin + 1; off < off_end; off++) {
                    auto &left_seg = data[off - 1];
                    auto &curr_seg = data[off];
                    if (left_seg.label == 0xFFFF && curr_seg.label != 0xFFFF) {
                        local_merger->update(off - 1, off, curr_seg.label);
                    } else if (left_seg.label != 0xFFFF && curr_seg.label == 0xFFFF) {
                        local_merger->update(off, off - 1, left_seg.label);
                    } else if (left_seg.label == 0xFFFF && curr_seg.label == 0xFFFF) {
                        local_merger->add_undef_neighbor(off - 1, off);
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
                    if (data[up_ix].x_end <= data[curr_ix].x) {
                        up_ix++;
                    } else if (data[curr_ix].x_end <= data[up_ix].x) {
                        curr_ix++;
                    } else {
                        // if control flows through here, it means prev and curr overlap
                        if (up_seg.label != 0xFFFF && curr_seg.label == 0xFFFF) {
                            local_merger->update(curr_ix, up_ix, up_seg.label);
                        } else if (up_seg.label == 0xFFFF && curr_seg.label != 0xFFFF) {
                            local_merger->update(up_ix, curr_ix, curr_seg.label);
                        } else if (up_seg.label == 0xFFFF && curr_seg.label == 0xFFFF) {
                            local_merger->add_undef_neighbor(up_ix, curr_ix);
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
        auto t1 = Clock::now();

        AdjMerger merger(segment_set, cc_set);
        for (const auto &local_merger : local_mergers) {
            merger.concat(*local_merger);
        }
        auto t2 = Clock::now();
        merger.group_neighbors();
        auto t3 = Clock::now();
        merger.copy_back(dest);
        auto t4 = Clock::now();

        std::cerr << "    adj.pre: " << micro(t0 - t01) << "us\n";
        std::cerr << "    adj.adjcalc: " << micro(t1 - t0) << "us\n";
        std::cerr << "    adj.mergecalc: " << micro(t2 - t1) << "us\n";
        std::cerr << "    adj.group: " << micro(t3 - t2) << "us\n";
        std::cerr << "    adj.copyback: " << micro(t4 - t3) << "us\n";
        std::cerr << "    adj.__all__: " << micro(t4 - t01) << "us\n";
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
                merge_segment_rows<false>(segment_set, dest, i);
            }

            #pragma omp critical
            seam_ys.push_back(seam);
        }
        for (int i : seam_ys) {
            if (i <= 0) continue;
            merge_segment_rows<true>(segment_set, dest, i);
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

    ConnectivityEnforcer::ConnectivityEnforcer(const uint16_t *labels, int H, int W, int K, int min_threshold)
            : min_threshold(min_threshold), max_label_size(K) {
        auto t0 = Clock::now();
        segment_set.set_from_2d_array(labels, H, W);
        auto t1 = Clock::now();
        std::cerr << "  row segmentation: " << micro(t1 -t0) << "us" << std::endl;
    }

    void ConnectivityEnforcer::execute(label_no_t *out) {
        auto t0 = Clock::now();
        DisjointSet disjoint_set;
        assign_disjoint_set(segment_set, disjoint_set);

        auto t1 = Clock::now();
        std::unique_ptr<ComponentSet> cc_set { disjoint_set.flatten() };
        std::vector<int> component_area;
        estimate_component_area(segment_set, *cc_set, component_area); // ComponentNo -> int (area)
        auto t3 = Clock::now();

        std::vector<component_no_t> largest_component(max_label_size, -1); // Label -> ComponentNo
        std::vector<int> largest_area(max_label_size, 0); // Label -> int

        #pragma omp parallel
        {
            int num_components = cc_set->get_num_components();
            auto &data = segment_set.get_mutable_data();
            std::vector<int> local_largest_area(max_label_size, 0); // Label -> int
            std::vector<component_no_t> local_largest_component(max_label_size, -1); // Label -> ComponentNo
            #pragma omp for
            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                segment_no_t segment_leader = cc_set->component_leaders[component_no];
                label_no_t label = data[segment_leader].label;
                if (label == 0xFFFF) continue;
                int area = component_area[component_no];
                if (area >= min_threshold && local_largest_area[label] < area) {
                    local_largest_area[label] = area;
                    local_largest_component[label] = component_no;
                }
            }

            #pragma omp critical
            for (int i = 0; i < max_label_size; i++) {
                if (largest_area[i] < local_largest_area[i]) {
                    largest_area[i] = local_largest_area[i];
                    largest_component[i] = local_largest_component[i];
                }
            }
            #pragma omp barrier

            #pragma omp for
            for (int ix = 0; ix < (int)data.size(); ix++) {
                if (largest_component[data[ix].label] != cc_set->component_assignment[ix]) {
                    data[ix].label = 0xFFFF;
                }
            }
        }

        auto t4 = Clock::now();
        std::vector<label_no_t> adj; // ComponentNo -> LabelNo
        unlabeled_adj(segment_set, *cc_set, adj);
        auto t5 = Clock::now();
        int width = segment_set.get_width(), height = segment_set.get_height();
        const auto &row_offsets = segment_set.get_row_offsets();
        const auto &data = segment_set.get_data();

        #pragma omp parallel for
        for (int i = 0; i < height; i++) {
            for (int off = row_offsets[i]; off < row_offsets[i + 1]; off++) {
                const RowSegment &segment = data[off];
                if (segment.label != 0xFFFF) continue;
                component_no_t component_no = cc_set->component_assignment[off];
                label_no_t label_subs = adj[component_no];
                if (label_subs == 0xFFFF) label_subs = 0;
                for (int j = segment.x; j < segment.x_end; j++) {
                    out[i * width + j] = label_subs;
                }
            }
        }
        auto t6 = Clock::now();

        std::cerr << "  disjoint: " << micro(t1 -t0) << "us" << std::endl;
        std::cerr << "  flatten: " << micro(t3 - t1) << "us" << std::endl;
        std::cerr << "  unlabel_comp: " << micro(t4 - t3) << "us" << std::endl;
        std::cerr << "  adj: " << micro(t5 - t4) << "us" << std::endl;
        std::cerr << "  writeback_comp: " << micro(t6 - t5) << "us" << std::endl;
    }
};
