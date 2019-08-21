#ifndef _FAST_SLIC_CCA_H
#define _FAST_SLIC_CCA_H

#include <cstdint>
#include <vector>
#include <memory>
#include <functional>

namespace cca {
    using label_no_t = uint16_t;
    using segment_no_t = int;
    using component_no_t = int;
    using tree_node_t = int;

    struct RowSegment {
        label_no_t label;
        int16_t y;
        int16_t x;
        int16_t x_end;
        bool mergable;
        RowSegment() {};
        RowSegment(label_no_t label, int16_t y, int16_t x, int16_t x_end) :
            label(label), y(y), x(x), x_end(x_end), mergable(false) {};
        int16_t get_length() const { return x_end - x; };
    };

    class kernel_function {
    public:
        virtual int get_ndims() = 0;
        virtual void operator() (RowSegment**, int, label_no_t, float*, float*) = 0;
        virtual ~kernel_function() {};
    };

    class ComponentSet;
    class RowSegmentSet {
    private:
        std::vector<RowSegment> data;
        std::vector<int> row_offsets;
        int width;
        int first_row_idx;
    public:
        RowSegmentSet() { clear(); };
        RowSegmentSet& operator+=(const RowSegmentSet &rhs);
        void clear() { row_offsets.clear(); row_offsets.push_back(0); data.clear(); };
        void set_from_2d_array(const label_no_t *labels, int H, int W);
        int get_height() const { return (int)row_offsets.size() - 1; };
        int get_width() const { return width; };
        int size() const { return (int)data.size(); };

        const std::vector<RowSegment>& get_data() const { return data; };
        std::vector<RowSegment>& get_mutable_data() { return data; };
        const std::vector<int>& get_row_offsets() const { return row_offsets; };

        void collapse();
    };

    class DisjointSet {
    public:
        int size;
        std::vector<tree_node_t> parents; // TreeNodeNo -> TreeNodeNo (of parent)
    public:
        DisjointSet() : size(0) {};
        DisjointSet(int size) : size(size), parents(size) {
            for (tree_node_t i = 0; i < size; i++) {
                parents[i] = i;
            }
        }

    private:
        inline void set_root(tree_node_t root, tree_node_t node) {
            for (tree_node_t i = node; root < i; ) {
                int parent = parents[i];
                parents[i] = root;
                i = parent;
            }
        }
    public:
        inline tree_node_t add() {
            tree_node_t c = size++;
            parents.push_back(c);
            return c;
        }
        inline void clear() {
            parents.clear();
            size = 0;
        }

        inline int find_root(tree_node_t node) {
            tree_node_t parent = parents[node];
            while (parent < node) {
                node = parent;
                parent = parents[parent];
            }
            return node;
        }
        inline tree_node_t find(tree_node_t node) {
            tree_node_t root = find_root(node);
            set_root(root, node);
            return root;
        }

        inline tree_node_t merge(tree_node_t node_i, tree_node_t node_j) {
            tree_node_t root = find_root(node_i);
            tree_node_t root_j = find_root(node_j);
            if (root > root_j) root = root_j;
            set_root(root, node_j);
            set_root(root, node_i);
            return root;
        }

        inline void add_single(tree_node_t node_i, tree_node_t single_j) {
            parents[single_j] = parents[node_i];
        }
        std::unique_ptr<ComponentSet> flatten();
    };


    class ComponentSet {
    public:
        int num_components;
        std::vector<component_no_t> component_assignment; // SegmentNo-> ComponentNo
        std::vector<int> num_component_members; // ComponentNo -> #OfMembers
        std::vector<segment_no_t> component_leaders; // ComponentNo -> SegmentIndex
    public:
        ComponentSet(int segment_size) : component_assignment(segment_size, -1) {};
        int get_num_components() const { return (int)num_component_members.size(); };
    };

    void assign_disjoint_set(const RowSegmentSet &segment_set, DisjointSet &dest);
    void estimate_component_area(const RowSegmentSet &segment_set, const ComponentSet &cc_set, std::vector<int> &dest);
    void unlabeled_adj(RowSegmentSet &segment_set, const ComponentSet &cc_set, kernel_function &kernel_fn, std::vector<label_no_t> &dest);


    class ConnectivityEnforcer {
    private:
        int min_threshold;
        int max_label_size;
        bool strict;
        std::vector<int> label_area_tbl;
        RowSegmentSet segment_set;
        kernel_function& kernel_fn;
    public:
        ConnectivityEnforcer(const uint16_t *labels, int H, int W, int K, int min_threshold, kernel_function &kernel_fn, bool strict = true);
        ConnectivityEnforcer(const uint16_t *labels, int H, int W, int K, int min_threshold, bool strict = true);
        void execute(label_no_t *out);
    };
};
#endif
