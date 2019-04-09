# cython: language_level=3
# distutils: sources = fast-slic.cpp

cimport cfast_slic

from libc.stdint cimport uint8_t
from libc.stdlib cimport malloc, free

cdef class SlicModel:
    cdef int num_components
    cdef cfast_slic.Cluster* _c_clusters

    def __cinit__(self, num_components):
        self.num_components = num_components
        self._c_clusters = <cfast_slic.Cluster *>malloc(sizeof(cfast_slic.Cluster) * num_components)

    cdef void initialize(self, const uint8_t [:, :, ::1] image) nogil:
        if image.shape[2] != 3:
            raise TypeError("nchan != 3")

        cdef int H = image.shape[0]
        cdef int W = image.shape[1]
        cdef int K = self.num_components

        if H > 0 and W > 0:
            cfast_slic.slic_initialize_clusters(H, W, K, &image[0, 0, 0], self._c_clusters)
        

    def __dealloc__(self):
        if self._c_clusters is not NULL:
            free(self._c_clusters)


cdef fast_slic():
    ...

