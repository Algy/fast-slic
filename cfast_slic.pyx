# cython: language_level=3, boundscheck=False
# distutils: sources = fast-slic.cpp

cimport cfast_slic
cimport numpy as np

import numpy as np

from libc.stdint cimport uint8_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

cdef class SlicCluster:
    def __init__(self, number, yx, color, num_members):
        self.number = number
        self.yx = yx
        self.color = color
        self.num_members = num_members




cdef class SlicModel:
    cdef int _num_components
    cdef cfast_slic.Cluster* _c_clusters
    cdef int _initialized

    def __cinit__(self, num_components):
        if num_components >= 65535:
            raise ValueError("num_components cannot exceed 65535")

        self._num_components = num_components
        self._c_clusters = <cfast_slic.Cluster *>malloc(sizeof(cfast_slic.Cluster) * num_components)
        self._initialized = False

    def copy(self):
        result = SlicModel(self._num_components)
        memcpy(result._c_clusters, self._c_clusters, sizeof(cfast_slic.Cluster) * self._num_components)
        return result

    @property
    def clusters(self):
        cdef int i
        cdef cfast_slic.Cluster* cluster

        result = []
        for i in range(0, self.num_components):
            cluster = self._c_clusters + i
            result.append(
                SlicCluster(
                    number=cluster.number,
                    yx=(cluster.y, cluster.x),
                    color=(cluster.r,  cluster.g, cluster.b),
                    num_members=cluster.num_members,
                )
            )
        return result


    cdef void _initialize(self, const uint8_t [:, :, ::1] image):
        if image.shape[2] != 3:
            raise ValueError("nchan != 3")

        cdef int H = image.shape[0]
        cdef int W = image.shape[1]
        cdef int K = self._num_components

        if H > 0 and W > 0:
            with nogil:
                cfast_slic.slic_initialize_clusters(H, W, K, &image[0, 0, 0], self._c_clusters)
        else:
            raise ValueError("image cannot be empty")
        self._initialized = True

    def initialize(self, image):
        return self._initialize(image)

    def initialized(self):
        return self._initialized == 1

    def num_components(self):
        return self._num_components

    def __dealloc__(self):
        if self._c_clusters is not NULL:
            free(self._c_clusters)


cdef class Slic:
    cdef SlicModel _slic_model
    cdef compactness_shift
    cdef quantize_level

    def __init__(self, num_components=None, slic_model=None, compactness_shift=5, quantize_level=7):
        self.compactness_shift = compactness_shift
        self.quantize_level = quantize_level
        self._slic_model = slic_model or SlicModel(num_components or 100)

    def num_components(self):
        return self._slic_model.num_components

    @property
    def slic_model(self):
        return self._slic_model.copy()

    @slic_model.setter
    def slic_model(self, v):
        self._slic_model = v

    cdef _iterate(self, const uint8_t [:, :, ::1] image, int max_iter):
        if image.shape[2] != 3:
            raise ValueError("nchan != 3")
        cdef int H = image.shape[0]
        cdef int W = image.shape[1]
        cdef int K = self._num_components
        cdef np.ndarray[np.uint32_t, ndim=2, mode='c'] assignments = np.zeros([H, W], dtype=np.uint32)

        
        cdef uint8_t quantize_level = self.quantize_level
        cdef uint8_t compactness_shift = self.compactness_shift
        cdef cfast_slic.Cluster * c_clusters = self._slic_model._c_clusters
        with nogil:
            cfast_slic.do_slic(
                H,
                W,
                K,
                compactness_shift,
                quantize_level,
                max_iter,
                &image[0, 0, 0],
                c_clusters,
                &assignments[0, 0]
            )

        assignments = assignments.astype(np.int32)
        assignments[assignments == 0xFFFF] = -1
        return assignments

    def iterate(self, image, max_iter=10):
        return self._iterate(image, max_iter)

