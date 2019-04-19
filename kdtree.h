#ifndef _KDTREE_H
#define _KDTREE_H

#include <algorithm>
#include <climits>
#include <vector>
#include <iostream>

namespace mykdtree {
    template <typename T>
    inline static T pow2(T a) {
        return a * a;
    }

    template <typename T>
    inline static T my_abs(T a) {
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

        inline int coord_of_dimension(int dimension) const { return coord.pt[dimension]; };
        inline int distance_to(const KDTreePoint &other) const { return my_abs(coord.pt[0] - other.coord.pt[0]) + my_abs(coord.pt[1] - other.coord.pt[1]); };
    };



    template <typename T>
    class KDTreeNode {
    public:
        const int dimension;
        const int value;

        // Borrowed from KDTree
        std::vector< KDTreePoint<T>* > point_ptrs;
        KDTreeNode<T> *lt_node;
        KDTreeNode<T> *gt_node;

        KDTreeNode(int dimension, int value) : dimension(dimension), value(value), lt_node(nullptr), gt_node(nullptr) {};

        void add(KDTreePoint<T> *ptr) { point_ptrs.push_back(ptr); };

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
    public:
        KDHeapItem(int distance, KDTreePoint<T> *point_ptr) : distance(distance), point_ptr(point_ptr) {};
    };

    template <typename T>
    bool operator<(const KDHeapItem<T>& lhs, const KDHeapItem<T>& rhs) {
        return lhs.distance < rhs.distance;
    }

    template <typename T>
    class KDTree {
        class kd_tree_sort_op {
        private:
            int dimension;
        public:
            kd_tree_sort_op(int dimension) : dimension(dimension) {};
            bool operator()(const KDTreePoint<T> *lhs, const KDTreePoint<T> *rhs) {
                return lhs->coord.pt[dimension] < rhs->coord.pt[dimension];
            };
        };

    private:
        KDTreeNode<T> *root;
        std::vector< KDTreePoint<T> > points;
    public:
        KDTree() : root(nullptr) {};
        KDTree(const KDTree<T>&other) = delete;
        KDTree& operator=(const KDTree<T>& other) = delete;

        void push_back(int x, int y, T data) {
            points.push_back(KDTreePoint<T>(x, y, data));
        }

        void bulk_build() {
            if (root != nullptr) delete root;
            KDTreePoint<T>** point_ptrs = new KDTreePoint<T>*[points.size()];
            for (size_t i = 0; i < points.size(); i++) {
                point_ptrs[i] = &points[i];
            }
            root = construct(point_ptrs, points.size(), 0);
            delete [] point_ptrs;
        }

        std::vector<KDTreePoint<T>*> k_nearest_neighbors(int x, int y, size_t k) {
            KDTreePoint<T> point(x, y);
            std::vector<KDHeapItem<T>> heap;
            knn_search(&point, root, heap, 0, k);
            std::sort_heap(heap.begin(), heap.end());

            std::vector<KDTreePoint<T>*> results;
            for (KDHeapItem<T> &item : heap) {
                results.push_back(item.point_ptr);
            }

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
            int median_coord = point_ptrs[median_index]->coord_of_dimension(dimension);
            for (; left_median_index >= 0 && point_ptrs[left_median_index]->coord_of_dimension(dimension) == median_coord; left_median_index--);
            for (; right_median_index < (int)length && point_ptrs[right_median_index]->coord_of_dimension(dimension) == median_coord; right_median_index++);

            KDTreeNode<T>* node = new KDTreeNode<T>(dimension, median_coord);
            for (int i = left_median_index + 1; i <= right_median_index - 1; i++) {
                node->add(point_ptrs[i]);
            }

            const int next_dimension = (dimension + 1) % 2;
            node->lt_node = construct(point_ptrs, (size_t)(left_median_index + 1), next_dimension);
            node->gt_node = construct(point_ptrs + right_median_index, length - right_median_index, next_dimension);
            return node;
        }

        void knn_search(const KDTreePoint<T>* point, const KDTreeNode<T>* node, std::vector<KDHeapItem<T>> &heap, const int dimension, const size_t k) {
            if (node == nullptr) return;
            for (KDTreePoint<T>* pivot_point_ptr : node->point_ptrs) {
                int distance = point->distance_to(*pivot_point_ptr);
                if (!heap.empty() && heap.front().distance <= distance) continue;
                heap.push_back(KDHeapItem<T>(distance, pivot_point_ptr));
                std::push_heap(heap.begin(), heap.end());
                while (heap.size() > k) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.pop_back();
                }
            }

            const int point_coord = point->coord_of_dimension(dimension);
            const int pivot_coord = node->value;
            const int next_dimension = (dimension + 1) % 2;

            int max_possible_distance;
            if (point_coord <= pivot_coord) {
                knn_search(point, node->lt_node, heap, next_dimension, k);
                if (heap.empty()) {
                    max_possible_distance = INT_MAX >> 2;
                } else {
                    max_possible_distance = heap.front().distance;
                }

                if (pivot_coord - max_possible_distance < point_coord) {
                    knn_search(point, node->gt_node, heap, next_dimension, k);
                }
            } else {
                knn_search(point, node->gt_node, heap, next_dimension, k);
                if (heap.empty()) {
                    max_possible_distance = INT_MAX >> 2;
                } else {
                    max_possible_distance = heap.front().distance;
                }

                if (pivot_coord + max_possible_distance > point_coord) {
                    knn_search(point, node->lt_node, heap, next_dimension, k);
                }
            }
        }
    };
}

#endif
