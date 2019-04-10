# cython: language_level=3

cimport cfast_slic 

from libc.stdint cimport uint8_t, uint32_t, uint16_t

cdef extern from "fast-slic.h":
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

    void slic_assign(int H, int W, int K, uint8_t compactness_shift, uint8_t quantize_level, const uint8_t* image, const Cluster* clusters, uint32_t* assignment) nogil
    void slic_update_clusters(int H, int W, int K, const uint8_t* image, Cluster* clusters, const uint32_t* assignment) nogil
    void slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters) nogil
    void do_slic(int H, int W, int K, uint8_t compactness_shift, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint32_t* assignment) nogil


cdef class SlicModel:
    cdef cfast_slic.Cluster* _c_clusters
    cdef readonly int num_components
    cdef public object initialized


    cpdef void initialize(self, const uint8_t [:, :, ::1] image)
    cpdef iterate(self, const uint8_t [:, :, ::1] image, int max_iter, uint8_t compactness_shift, uint8_t quantize_level)
    cdef _get_clusters(self)
