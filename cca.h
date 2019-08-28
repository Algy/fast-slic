#include <cstdint>
#include <vector>
#include <memory>

namespace cca {
    using label_no_t = uint16_t;
    using segment_no_t = int;
    using component_no_t = int;
    using tree_node_t = int;

    class DisjointSet;
    class ComponentSet;

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

    class ConnectivityEnforcer {
    private:
        int H, W;
        int min_threshold;
        int max_label_size;
        bool strict;
    public:
        ConnectivityEnforcer(const uint16_t *labels, int H, int W, int K, int min_threshold, bool strict = true);
        void execute(label_no_t *out);
    };
};
