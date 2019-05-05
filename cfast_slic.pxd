# cython: language_level=3

from libc.stdint cimport uint8_t, uint32_t, uint16_t, int16_t

cdef extern from "fast-slic-common.h":
    ctypedef struct Cluster:
        uint16_t y
        uint16_t x;
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t reserved[1]
        uint16_t number
        uint8_t is_active
        uint32_t num_members

    ctypedef struct Connectivity:
        int num_nodes;
        int *num_neighbors;
        uint32_t **neighbors;


cdef extern from "fast-slic.h":
    void fast_slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters) nogil
    void fast_slic_iterate(int H, int W, int K, float compactness, float min_size_factor, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint16_t* assignment) nogil
    Connectivity* fast_slic_get_connectivity(int H, int W, int K, const uint16_t *assignment) nogil
    Connectivity* fast_slic_knn_connectivity(int H, int W, int K, const Cluster* clusters, int num_neighbors) nogil
    void fast_slic_free_connectivity(Connectivity* conn) nogil
    void fast_slic_get_mask_density(int H, int W, int K, const Cluster* clusters, const uint16_t* assignment, const uint8_t *mask, uint8_t *cluster_densities) nogil
    void fast_slic_cluster_density_to_mask(int H, int W, int K, const Cluster *clusters, const uint16_t* assignment, const uint8_t *cluster_densities, uint8_t *result) nogil

cdef extern from "fast-slic-avx2.h":
    void fast_slic_initialize_clusters_avx2(int H, int W, int K, const uint8_t* image, Cluster *clusters) nogil
    void fast_slic_iterate_avx2(int H, int W, int K, float compactness, float min_size_factor, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint16_t* assignment) nogil
    int fast_slic_supports_avx2() nogil


cdef class NodeConnectivity:
    cdef Connectivity* _c_connectivity
    cpdef tolist(self)

    @staticmethod
    cdef create(Connectivity* conn)


cdef class BaseSlicModel:
    cdef Cluster* _c_clusters
    cdef readonly int num_components
    cdef public object initialized

    cpdef void initialize(self, const uint8_t [:, :, ::1] image)
    cpdef iterate(self, const uint8_t [:, :, ::1] image, int max_iter, float compactness, float min_size_factor, uint8_t quantize_level)
    cpdef get_connectivity(self, const int16_t[:,::1] assignments)
    cpdef get_knn_connectivity(self, const int16_t[:,::1] assignments, size_t num_neighbors)
    cpdef get_mask_density(self, const uint8_t[:, ::1] mask, const int16_t[:, ::1] assignments)
    cpdef broadcast_density_to_mask(self, const uint8_t[::1] densities, const int16_t[:, ::1] assignments);
    cdef _get_clusters(self)
    cdef _set_clusters(self, clusters)

    cpdef _get_name(self)



cdef class SlicModel(BaseSlicModel):
    pass

cdef class SlicModelAvx2(BaseSlicModel):
    pass

