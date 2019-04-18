#include <algorithm>
#include <climit>
#include <vector>

namespace mykdtree {
    template <typename T>
    static T pow2(T a) {
        return a * a;
    }

    template <typename T>
    struct KDTreePoint {
        union {
            struct {
                int x;
                int y;
            } xy;
            int pt[2];
        } coord;
        T data;
        KDTreePoint(int x, int y)  {
            coord.xy.x = x;
            coord.xy.y = y;
        }

        KDTreePoint(int x, int y, T data) : data(data) {
            coord.xy.x = x;
            coord.xy.y = y;
        }
        int value_of_dimension(int dimension) { return coord.pt[dimension]; };
        int distance_to(const KDTreePoint &other) { return pow2(coord.pt[0] - other.coord.pt[0]) + pow2(coord.pt[1] - other.coord.pt[1]); };
    };



    template <typename T>
    class KDTreeNode {
    public:
        const int dimension;
        const int value;
        // Borrowed from KDTree
        std::vector<KDTreePoint<T> *> point_ptrs;
        KDTreeNode *lt_node;
        KDTreeNode *gt_node;

        KDTreeNode(int dimension, int value) : dimension(dimension), value(value), lt_node(nullptr), gt_node(nullptr) { };

        void add(KDTreePoint<T> *ptr) {
            point_ptrs.push_back(ptr);
        }
        bool is_leaf() { return lt_node == nullptr && gt_node == nullptr; };

        ~KDTreeNode() {
            if (lt_node != nullptr) {
                delete lt_node;
            }
            if (gt_node != nullptr) {
                delete gt_node;
            }
        }
    };

    template <typename T>
    class KDHeapItem {
    public:
        int distance;
        KDTreePoint<T> *point_ptr;
        friend class KDHeapItem;

    public:
        KDHeapItem(int distance, KDTreeNode<T> *point_ptr) : distance(distance), point_ptr(point_ptr) {};
    };

    bool operator<(const KDHeapItem& lhs, const KDHeapItem& rhs) {
        return lhs.distance < rhs.distance;
    }


    template <typename T>
    class KDTree {
        class kd_tree_sort_op {
        private:
            int dimension;
        public:
            kd_tree_sort_op(int dimension) : dimension(dimension) {};
            bool operator()(const KDTreePoint<T> *&lhs, const KDTreePoint<T> *&rhs) {
                return lhs->coord.pt[dimension] < rhs->coord.pt[dimension];
            };
        };

    private:
        KDTreeNode *root;
        std::vector< KDTreePoint<T> > points;
    public:
        KDTree() : root(nullptr) {};
        KDTree(const KDTree<T>&other) = delete;
        KDTree& operator=(const KDTree<T>& other) = delete;

        void push_back(int x, int y, T data) {
            points.push_back(KDTreePoint(x, y, data));
        }

        void bulk_build() {
            if (root != nullptr)
                delete root;
            KDTreePoint<T>** point_ptrs = new KDTreePoint<T>*[points.size()];
            for (size_t i = 0; i < points.size(); i++) {
                point_ptrs[i] = &points[i];
            }
            root = construct(point_ptrs, 0);
            delete [] point_ptrs;
        }

        std::vector<KDTreePoint<T>*> k_nearest_neighbor(int x, int y, size_t k) {
            std::vector<KDTreePoint<T>*> results;
            results.reserve(k);
            KDTreePoint point = K(x, y);
            std::vector<KDHeapItem<T>> heap;
            knn_search(point, root, heap, 0, k);
            std::transform(heap.begin(), heap.end(), results,  [](KDHeapItem<T> item) { return item.point_ptr; });
            return results;
        }

        ~KDTree() {
            if (root != nullptr)
                delete root;
        }
    private:
        KDTreeNode<T>* construct(KDTreePoint<T>** point_ptrs, size_t length, int dimension) {
            if (length == 0) return nullptr;
            std::sort(point_ptrs, point_ptrs + length, kd_tree_sort_op(dimension));

            const int median_index = (int)length / 2;
            int left_median_index = median_index - 1, right_median_index = median_index + 1;
            int median_value = point_ptrs[median_index]->value_of_dimension(dimension);
            for (; left_median_index >= 0 && point_ptrs[left_median_index]->value_of_dimension(dimension) == median_value; left_median_index--);
            for (; right_median_index < point_ptrs.size() && point_ptrs[right_median_index]->value_of_dimension(dimension) == median_value; right_median_index++);

            KDTreeNode<T>* node = new KDTreeNode<T>(dimension, median_value);
            for (int i = left_median_index + 1; i <= right_median_index - 1; i++) {
                node->add(point_ptrs[i]);
            }

            const int next_dimension = (dimension + 1) % 2;
            node->lt_node = construct(point_ptrs, left_median_index + 1);
            node->gt_node = construct(point_ptrs + right_median_index, length - right_median_index);
            return node;
        }

        void knn_search(const KDTreePoint* point, const KDTreeNode<T>* node, std::vector<KDHeapItem<T>> &heap, const int dimension, const size_t k) {
            if (node == nullptr) return;
            for (KDTreePoint<T>* pivot_point_ptr : node->point_ptrs) {
                heap.push_back(KDHeapItem<T>(point->distance_to(*pivot_point_ptr), node));
                std::push_heap(heap.begin(), heap.end());
                while (heap.size() > k) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.pop_back();
                }
            }

            const int point_value = point->value_of_dimension(dimension);
            const int pivot_value = node->value;

            int current_distance;
            if (heap.empty()) {
                current_distance = INT_MAX >> 2;
            } else {
                current_distance = heap.front().distance;
            }

            int next_dimension = (dimension + 1) % 2;
            if (point_value <= pivot_value - current_distance) {
                knn_search(point, node->lt_node, heap, next_dimension, k);
            } else if (point_value >= pivot_value + current_distance) {
                knn_search(point, node->gt_node, heap, next_dimension, k);
            } else {
                knn_search(point, node->lt_node, heap, next_dimension, k);
                knn_search(point, node->gt_node, heap, next_dimension, k);
            }
        }
    };
}
