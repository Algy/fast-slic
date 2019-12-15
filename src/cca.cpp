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
#include <deque>
#include "timer.h"
#include "parallel.h"

typedef std::chrono::high_resolution_clock Clock;

template <typename T>
static int micro(T o) {
    return std::chrono::duration_cast<std::chrono::microseconds>(o).count();
}

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif

namespace cca {
    DisjointSet assign_disjoint_set(const label_no_t* assignment, int H, int W) {
        DisjointSet cc_set(H * W);

        std::vector<int> seam_ys;
        #pragma omp parallel num_threads(fsparallel::nth())
        {
            bool is_first = true;
            int seam = 0;
            #pragma omp for
            for (int i = 0; i < H; i++) {
                if (is_first) {
                    is_first = false;
                    seam = i;
                    label_no_t left_cluster_no = assignment[i * W];
                    for (int j = 1; j < W; j++) {
                        int index = i * W + j;
                        label_no_t cluster_no = assignment[index];
                        if (left_cluster_no == cluster_no) {
                            cc_set.merge(index - 1, index);
                        }
                        left_cluster_no = cluster_no;
                    }
                    continue;
                }

                label_no_t left_cluster_no;
                {
                    int index = i * W;
                    int up_index = index - W;
                    label_no_t cluster_no = assignment[index];
                    if (assignment[up_index] == cluster_no) {
                        cc_set.merge(up_index, index);
                    }
                    left_cluster_no = cluster_no;
                }
                for (int j = 1; j < W; j++) {
                    int index = i * W + j;
                    label_no_t cluster_no = assignment[index];
                    int left_index = index - 1, up_index = index - W;

                    if (left_cluster_no == cluster_no) {
                        cc_set.merge(left_index, index);
                        if (assignment[up_index] == cluster_no) {
                            cc_set.merge(left_index, up_index);
                        }
                    } else if (assignment[up_index] == cluster_no) {
                        cc_set.merge(up_index, index);
                    }
                    left_cluster_no = cluster_no;
                }
            }

            #pragma omp critical
            seam_ys.push_back(seam);
        }

        for (int i : seam_ys) {
            if (i <= 0) continue;
            for (int j = 0; j < W; j++) {
                int index = i * W + j;
                int up_index = index - W;
                label_no_t cluster_no = assignment[index];
                if (assignment[up_index] == cluster_no) {
                    cc_set.merge(index, up_index);
                }
            }
        }
        return cc_set;
    }

    std::unique_ptr<ComponentSet> DisjointSet::flatten() {
        int size = (int)parents.size();
        std::unique_ptr<ComponentSet> result { new ComponentSet(size) };
        std::vector<std::vector<tree_node_t>> rootset;
        std::vector<int> root_offsets;
        #pragma omp parallel num_threads(fsparallel::nth())
        {
            #pragma omp single
            {
                rootset.resize(omp_get_num_threads());
                root_offsets.resize(omp_get_num_threads() + 1, 0);
            }

            auto &local_roots = rootset[omp_get_thread_num()];
            // First, rename leading nodes
            #pragma omp for
            for (tree_node_t i = 0; i < size; i++) {
                if (parents[i] == i) {
                    local_roots.push_back(i);
                }
            }

            #pragma omp single
            for (int i = 0; i < (int)rootset.size(); i++) {
                root_offsets[i + 1] = root_offsets[i] + (int)rootset[i].size();
            }
            int local_root_offset = root_offsets[omp_get_thread_num()];

            tree_node_t component_counter = 0;
            for (tree_node_t i : local_roots) {
                result->component_assignment[i] = local_root_offset + component_counter++;
            }

            // Second, allocate info arrays
            #pragma omp single
            {
                result->num_components = root_offsets.back();
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
            : H(H), W(W), min_threshold(min_threshold), max_label_size(K) {};

    void ConnectivityEnforcer::execute(label_no_t *out) {
        struct areacmpcls {
            const std::vector<int> &area_tbl;
            areacmpcls(const std::vector<int> & area_tbl) : area_tbl(area_tbl) {};
            bool operator()(const component_no_t lhs, const component_no_t rhs) {
                return area_tbl[lhs] > area_tbl[rhs];
            }
        };
        struct leader_index_cmpcls {
            const std::vector<segment_no_t> &component_leaders;
            leader_index_cmpcls(const std::vector<int> & component_leaders) : component_leaders(component_leaders) {};
            bool operator()(const component_no_t lhs, const component_no_t rhs) {
                return component_leaders[lhs] < component_leaders[rhs];
            }
        };

        fstimer::Scope s("cca");
        std::unique_ptr<ComponentSet> cc_set;
        {
            fstimer::begin("build_disjoint_set");
            DisjointSet disjoint_set = assign_disjoint_set(out, H, W);
            fstimer::end();
            fstimer::begin("flatten");
            cc_set = std::move(disjoint_set.flatten());
            fstimer::end();
        }

        int num_components = cc_set->num_components;

        std::vector<label_no_t> substitute(num_components, 0xFFFF);
        std::vector<component_no_t> comps;
        comps.reserve(max_label_size);

        {
            fstimer::Scope s("threshold_by_area");
            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                if (cc_set->num_component_members[component_no] >= min_threshold) {
                    comps.push_back(component_no);
                }
            }
        }

        areacmpcls areacmp(cc_set->num_component_members);
        leader_index_cmpcls leadercmp(cc_set->component_leaders);

        {
            fstimer::Scope s("sort");
            if ((std::size_t)max_label_size < comps.size()) {
                std::partial_sort(comps.begin(), comps.begin() + max_label_size, comps.end(), areacmp);
                comps.erase(comps.begin() + max_label_size, comps.end());
            }
            std::sort(comps.begin(), comps.end(), leadercmp);
        }
        label_no_t next_label = 0;

        {
            fstimer::Scope s("substitute");
            for (component_no_t component_no : comps) {
                substitute[component_no] = next_label++;
            }
            if (num_components > 0 && substitute[0] == 0xFFFF) substitute[0] = 0;

            for (component_no_t component_no = 0; component_no < num_components; component_no++) {
                if (substitute[component_no] != 0xFFFF) continue;
                int leader_index = cc_set->component_leaders[component_no];
                label_no_t subs_label = 0xFFFF;
                if (leader_index % W > 0) {
                    subs_label = substitute[cc_set->component_assignment[leader_index - 1]];
                } else {
                    subs_label = substitute[cc_set->component_assignment[leader_index - W]];
                }
                if (subs_label == 0xFFFF) {
                    subs_label = 0;
                    // std::cerr << "leader_y " << leader_index << "\n";
                }
                substitute[component_no] = subs_label;
            }
        }


        {
            fstimer::Scope s("output");
            #pragma omp parallel for num_threads(fsparallel::nth())
            for (int i = 0; i < H * W; i++) {
                out[i] = substitute[cc_set->component_assignment[i]];
            }
        }
    }
};
