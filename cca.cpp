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


namespace cca {
    struct component_pair {
        component_no_t component_no;
        int area;
        component_pair(component_no_t component_no, int area) : component_no(component_no), area(area) {};
    };

    struct component_pair_cmp {
        bool operator()(const component_pair &lhs, const component_pair &rhs) {
            return lhs.area > rhs.area;
        }
    };

    class AdjMerger {
    private:
        const RowSegmentSet &segment_set;
        const ComponentSet &cc_set;
        const int* component_area; // CompoenntNo -> int
        std::vector<int> largest_adj_area; // ComponentNo -> area
        std::vector<component_no_t> largest_adj_comp; // ComponentNo -> Label
        int max_label_size;
        int min_threshold;
    public:
        int size() { return largest_adj_area.size(); };
        AdjMerger(const RowSegmentSet &segment_set, const ComponentSet &cc_set, const int* component_area, int max_label_size, int min_threshold) :
                segment_set(segment_set), cc_set(cc_set), component_area(component_area),
                largest_adj_area(cc_set.get_num_components(), std::numeric_limits<int>::min()),
                largest_adj_comp(cc_set.get_num_components(), -1),
                max_label_size(max_label_size), min_threshold(min_threshold) {};
        void update(segment_no_t target_ix, segment_no_t adj_ix) {
            component_no_t target = cc_set.component_assignment[target_ix];
            component_no_t adj = cc_set.component_assignment[adj_ix];

            int &target_largest_area = largest_adj_area[target];
            int adj_area = component_area[adj];
            if (target_largest_area < adj_area) {
                target_largest_area = adj_area;
                largest_adj_comp[target] = adj;
            }
        };

        void concat(const std::vector< std::unique_ptr<AdjMerger> >& local_mergers) {
            int num_components = cc_set.get_num_components();
            #pragma omp parallel
            {
                #pragma omp for
                for (int i = 0; i < num_components; i++) {
                    int max_area = std::numeric_limits<int>::min();
                    component_no_t max_comp = -1;
                    for (const auto &local_merger : local_mergers) {
                        int curr_area = local_merger->largest_adj_area[i];
                        if (max_area < curr_area) {
                            max_area = curr_area;
                            max_comp = local_merger->largest_adj_comp[i];
                        }
                    }
                    largest_adj_area[i] = max_area;
                    largest_adj_comp[i] = max_comp;
                }
            }
        };

        void copy_back(std::vector<label_no_t> &dest) {
            int num_components = cc_set.num_components;
            dest.resize(num_components);
            std::vector<int> mergable_map(num_components, 0);
            const auto &data = segment_set.get_data();

            int num_mergables = 0;

            label_no_t last_new_label = 0;
            std::vector<component_pair> recovered_component_pairs;
            #pragma omp parallel
            {
                #pragma omp for
                for (int i = 0; i < num_components; i++) {
                    if (data[cc_set.component_leaders[i]].label == 0xFFFF) {
                        mergable_map[i] = 1;
                    }
                }

                #pragma omp for reduction(+:num_mergables)
                for (int i = 0; i < num_components; i++) {
                    num_mergables += mergable_map[i];
                }

                #pragma omp single
                {
                    for (int i = 0; i < num_components; i++) {
                        if (!mergable_map[i]) {
                            dest[i] = last_new_label++;
                        }
                    }
                }

                int recovery_capacity = max_label_size - num_mergables;
                if (recovery_capacity > 0) {
                    std::vector<component_pair> local_queue;
                    #pragma omp for
                    for (int i = 0; i < num_components; i++) {
                        if (mergable_map[i]) {
                            local_queue.push_back(component_pair(i, component_area[i]));
                            std::push_heap(local_queue.begin(), local_queue.end(), component_pair_cmp());
                            if (local_queue.size() > recovery_capacity) {
                                std::pop_heap(local_queue.begin(), local_queue.end(), component_pair_cmp());
                                local_queue.pop_back();
                            }
                        }
                    }
                    #pragma omp critical
                    recovered_component_pairs.insert(
                        recovered_component_pairs.end(),
                        local_queue.begin(),
                        local_queue.end()
                    );
                    #pragma omp barrier
                    #pragma omp single
                    {
                        std::nth_element(
                            recovered_component_pairs.begin(),
                            recovered_component_pairs.begin() + recovery_capacity,
                            recovered_component_pairs.end(),
                            component_pair_cmp()
                        );

                        auto end = recovered_component_pairs.begin() + recovery_capacity;
                        for (auto it = recovered_component_pairs.begin(); it != end; ++it) {
                            if (it->area < min_threshold) break;
                            dest[it->component_no] = last_new_label++;
                            mergable_map[it->component_no] = false;
                        }
                    }
                }

                #pragma omp for
                for (int i = 0; i < num_components; i++) {
                    if (mergable_map[i]) {
                        component_no_t adj_comp = largest_adj_comp[i];
                        if (adj_comp != -1) {
                            dest[i] = dest[adj_comp];
                        } else {
                            // if (segment_set.get_data()[cc_set.component_leaders[i]].get_length() != 0)
                            //    abort();
                            dest[i] = 0xFFFF;
                        }
                    }
                }
            }
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


    void unlabeled_adj(const RowSegmentSet &segment_set, const ComponentSet &cc_set, const std::vector<int> &component_area, int max_label_size, int min_threshold, std::vector<label_no_t> &dest) {
        // auto t01 = Clock::now();
        int num_components = cc_set.get_num_components();
        const auto &row_offsets = segment_set.get_row_offsets();
        const auto &data = segment_set.get_data();
        auto height = segment_set.get_height();

        std::vector< std::unique_ptr<AdjMerger> > local_mergers;
        // left to right
        // auto t0 = Clock::now();
        #pragma omp parallel
        {
            AdjMerger *local_merger = new AdjMerger(segment_set, cc_set, &component_area[0], max_label_size, min_threshold);
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
                        local_merger->update(off - 1, off);
                    } else if (left_seg.label != 0xFFFF && curr_seg.label == 0xFFFF) {
                        local_merger->update(off, off - 1);
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
                            local_merger->update(curr_ix, up_ix);
                        } else if (up_seg.label == 0xFFFF && curr_seg.label != 0xFFFF) {
                            local_merger->update(up_ix, curr_ix);
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
        // // auto t1 = Clock::now();

        std::unique_ptr<AdjMerger> merger { new AdjMerger(segment_set, cc_set, &component_area[0], max_label_size, min_threshold) };
        merger->concat(local_mergers);

        // auto t2 = Clock::now();
        // auto t3 = Clock::now();
        merger->copy_back(dest);
        // auto t4 = Clock::now();
        // std::cerr << "    adj(__size__): " << merger->size() << " / " << num_components  << "\n";
        merger = nullptr;
        local_mergers.clear();
        // auto t5 = Clock::now();

        // std::cerr << "    adj.pre: " << micro(t0 - t01) << "us\n";
        // std::cerr << "    adj.adjcalc: " << micro(t1 - t0) << "us\n";
        // std::cerr << "    adj.mergecalc: " << micro(t2 - t1) << "us\n";
        // std::cerr << "    adj.group: " << micro(t3 - t2) << "us\n";
        // std::cerr << "    adj.copyback: " << micro(t4 - t3) << "us\n";
        // std::cerr << "    adj._del__: " << micro(t5 - t4) << "us\n";
        // std::cerr << "    adj.__all__: " << micro(t5 - t01) << "us\n";
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

    ConnectivityEnforcer::ConnectivityEnforcer(const uint16_t *labels, int H, int W, int K, int min_threshold, bool strict)
            : min_threshold(min_threshold), max_label_size(K), strict(strict) {
        // auto t0 = Clock::now();
        segment_set.set_from_2d_array(labels, H, W);
        // auto t1 = Clock::now();
        // std::cerr << "  row segmentation: " << micro(t1 -t0) << "us" << std::endl;
    }

    void ConnectivityEnforcer::execute(label_no_t *out) {

        // auto t0 = Clock::now();
        DisjointSet disjoint_set;
        assign_disjoint_set(segment_set, disjoint_set);

        // auto t1 = Clock::now();
        std::unique_ptr<ComponentSet> cc_set { disjoint_set.flatten() };
        std::vector<int> component_area;
        estimate_component_area(segment_set, *cc_set, component_area); // ComponentNo -> int (area)
        // auto t3 = Clock::now();

        int W = segment_set.get_width(), H = segment_set.get_height();
        int num_components = cc_set->num_components;
        std::vector<component_pair> cpairs;
        cpairs.reserve(num_components);
        for (component_no_t component_no = 0; component_no < num_components; component_no++) {
            cpairs.push_back(component_pair(component_no, component_area[component_no]));
        }

        if (strict) {
            auto &data = segment_set.get_mutable_data();
            std::vector<bool> cutoff_map(num_components, false);
            int cutoff = num_components < max_label_size? num_components : max_label_size;
            std::nth_element(cpairs.begin(), cpairs.begin() + cutoff, cpairs.end(), component_pair_cmp());
            for (auto it = cpairs.begin() + cutoff; it != cpairs.end(); ++it) {
                cutoff_map[it->component_no] = true;
            }
            for (auto it = cpairs.begin(); it != cpairs.begin() + cutoff; ++it) {
                if (it->area < min_threshold) {
                    cutoff_map[it->component_no] = true;
                }
            }
            #pragma omp parallel for
            for (int ix = 0; ix < data.size(); ix++) {
                if (cutoff_map[cc_set->component_assignment[ix]]) {
                    data[ix].label = 0xFFFF;
                }
            }
        } else {
            auto &data = segment_set.get_mutable_data();
            #pragma omp parallel for
            for (int ix = 0; ix < data.size(); ix++) {
                if (component_area[cc_set->component_assignment[ix]] < min_threshold) {
                    data[ix].label = 0xFFFF;
                }
            }
        }

        segment_set.collapse();

        // auto t4 = Clock::now();
        std::vector<label_no_t> adj; // ComponentNo -> LabelNo
        disjoint_set.clear();
        component_area.clear();

        assign_disjoint_set(segment_set, disjoint_set);
        cc_set = disjoint_set.flatten();
        estimate_component_area(segment_set, *cc_set, component_area);
        std::vector<bool> mergable_map(num_components, false);
        unlabeled_adj(segment_set, *cc_set, component_area, max_label_size, min_threshold, adj);

        // auto t5 = Clock::now();
        int width = segment_set.get_width(), height = segment_set.get_height();
        const auto &row_offsets = segment_set.get_row_offsets();
        const auto &data = segment_set.get_data();
        #pragma omp parallel for
        for (int i = 0; i < height; i++) {
            for (int off = row_offsets[i]; off < row_offsets[i + 1]; off++) {
                const RowSegment &segment = data[off];
                component_no_t component_no = cc_set->component_assignment[off];
                label_no_t label_subs = adj[component_no];
                if (label_subs == 0xFFFF) label_subs = 0;
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
