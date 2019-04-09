# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t


cdef extern from "fast-slic.h":
    ctypedef struct Cluster:
        pass
    void slic_assign(int H, int W, int K, uint8_t compactness_shift, uint8_t quantize_level, const uint8_t* image, const Cluster* clusters, uint32_t* assignment) nogil
    void slic_update_clusters(int H, int W, int K, const uint8_t* image, Cluster* clusters, const uint32_t* assignment) nogil
    void slic_initialize_clusters(int H, int W, int K, const uint8_t* image, Cluster *clusters) nogil
    void do_slic(int H, int W, int K, uint8_t compactness_shift, uint8_t quantize_level, int max_iter, const uint8_t* image, Cluster* clusters, uint32_t* assignment) nogil
