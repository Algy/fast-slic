# cython: language_level=3

from libc.stdint cimport uint8_t, uint32_t, uint16_t

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

cdef extern from "fast-slic.h":
    void fast_slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters) nogil
    void fast_slic_iterate(int H, int W, int K, uint8_t compactness, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint32_t* assignment) nogil

cdef extern from "fast-slic-avx2.h":
    void fast_slic_initialize_clusters_avx2(int H, int W, int K, const uint8_t* image, Cluster *clusters) nogil
    void fast_slic_iterate_avx2(int H, int W, int K, uint8_t compactness, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint32_t* assignment) nogil
    int fast_slic_supports_avx2() nogil


cdef class BaseSlicModel:
    cdef Cluster* _c_clusters
    cdef readonly int num_components
    cdef public object initialized

    cpdef void initialize(self, const uint8_t [:, :, ::1] image)
    cpdef iterate(self, const uint8_t [:, :, ::1] image, int max_iter, uint8_t compactness, uint8_t quantize_level)
    cdef _get_clusters(self)
    cpdef _get_name(self)


cdef class SlicModel(BaseSlicModel):
    pass

cdef class SlicModelAvx2(BaseSlicModel):
    pass

