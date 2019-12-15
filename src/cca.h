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
        inline tree_node_t add() {
            tree_node_t c = size++;
            parents.push_back(c);
            return c;
        }

        inline void clear() {
            parents.clear();
            size = 0;
        }

        inline void merge(tree_node_t node_i, tree_node_t node_j) {
            tree_node_t root_x = node_i, root_y = node_j;
            while (parents[root_x] != parents[root_y]) {
                if (parents[root_x] > parents[root_y]) {
                    if (root_x == parents[root_x]) {
                        parents[root_x] = parents[root_y];
                        break;
                    }
                    tree_node_t z = parents[root_x];
                    parents[root_x] = parents[root_y];
                    root_x = z;
                } else {
                    if (root_y == parents[root_y]) {
                        parents[root_y] = parents[root_x];
                        break;
                    }
                    tree_node_t z = parents[root_y];
                    parents[root_y] = parents[root_x];
                    root_y = z;
                }
            }
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
    public:
        ConnectivityEnforcer(const uint16_t *labels, int H, int W, int K, int min_threshold);
        void execute(label_no_t *out);
    };
};
