# cython: language_level=3, boundscheck=False

cimport cfast_slic 
cimport numpy as np

import numpy as np

from libc.stdint cimport uint8_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset


cdef class BaseSlicModel:
    def __cinit__(self, int num_components):
        if num_components >= 65535:
            raise ValueError("num_components cannot exceed 65535")
        elif num_components <= 0:
            raise ValueError("num_components should be a non-negative integer")

        self.num_components = num_components
        self._c_clusters = <cfast_slic.Cluster *>malloc(sizeof(cfast_slic.Cluster) * num_components)
        memset(self._c_clusters, 0, sizeof(cfast_slic.Cluster) * num_components)
        self.initialized = False

    def copy(self):
        result = SlicModel(self.num_components)
        memcpy(result._c_clusters, self._c_clusters, sizeof(cfast_slic.Cluster) * self.num_components)
        result.initialized = self.initialized
        return result


    cdef _get_clusters(self):
        cdef cfast_slic.Cluster* cluster
        cdef int i

        result = []
        for i in range(0, self.num_components):
            cluster = self._c_clusters + i
            result.append(
                dict(
                    number=cluster.number,
                    yx=(cluster.y, cluster.x),
                    color=(cluster.r,  cluster.g, cluster.b),
                    num_members=cluster.num_members,
                )
            )
        return result

    @property
    def clusters(self):
        return self._get_clusters()

    cpdef void initialize(self, const uint8_t [:, :, ::1] image):
        if image.shape[2] != 3:
            raise ValueError("nchan != 3")

        cdef int H = image.shape[0]
        cdef int W = image.shape[1]
        cdef int K = self.num_components

        if H > 0 and W > 0:
            if self._get_name() == "standard":
                cfast_slic.fast_slic_initialize_clusters(H, W, K, &image[0, 0, 0], self._c_clusters)
            elif self._get_name() == "avx2":
                cfast_slic.fast_slic_initialize_clusters_avx2(H, W, K, &image[0, 0, 0], self._c_clusters)
            else:
                raise RuntimeError("Not reachable")
        else:
            raise ValueError("image cannot be empty")
        self.initialized = True


    cpdef iterate(self, const uint8_t [:, :, ::1] image, int max_iter, uint8_t compactness, uint8_t quantize_level): 
        if not self.initialized:
            raise RuntimeError("Slic model is not initialized")
        if image.shape[2] != 3:
            raise ValueError("nchan != 3")
        cdef int H = image.shape[0]
        cdef int W = image.shape[1]
        cdef int K = self.num_components
        cdef np.ndarray[np.uint32_t, ndim=2, mode='c'] assignments = np.zeros([H, W], dtype=np.uint32)
        cdef cfast_slic.Cluster* c_clusters = self._c_clusters

        if self._get_name() == 'standard':
            cfast_slic.fast_slic_iterate(
                H,
                W,
                K,
                compactness,
                quantize_level,
                max_iter,
                &image[0, 0, 0],
                c_clusters,
                &assignments[0, 0]
            )
        elif self._get_name() == 'avx2':
            cfast_slic.fast_slic_iterate_avx2(
                H,
                W,
                K,
                compactness,
                quantize_level,
                max_iter,
                &image[0, 0, 0],
                c_clusters,
                &assignments[0, 0]
            )
        else:
            raise RuntimeError("Not reachable")
        result = assignments.astype(np.int32)
        result[result == 0xFFFF] = -1
        return result

    def __dealloc__(self):
        if self._c_clusters is not NULL:
            free(self._c_clusters)

    cpdef _get_name(self):
        raise NotImplementedError

cdef class SlicModel(BaseSlicModel):
    cpdef _get_name(self):
        return "standard"

cdef class SlicModelAvx2(BaseSlicModel):
    cpdef _get_name(self):
        return "avx2"


def slic_supports_arch(name):
    if name == 'avx2':
        return cfast_slic.fast_slic_supports_avx2() == 1
    return False

