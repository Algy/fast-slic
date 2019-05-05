# cython: language_level=3, boundscheck=False

cimport cfast_slic 
cimport numpy as np

import numpy as np

from libc.stdint cimport uint8_t, int32_t, uint32_t, uint16_t, int16_t
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

    cdef _set_clusters(self, clusters):
        cdef uint8_t r, g, b
        cdef uint16_t x, y
        cdef uint32_t num_members
        cdef int num_new_clusters, i
        cdef cfast_slic.Cluster* new_clusters

        num_new_clusters = len(clusters)
        new_clusters = <cfast_slic.Cluster *>malloc(sizeof(cfast_slic.Cluster) * num_new_clusters)
        try:
            for i in range(num_new_clusters):
                dict_ = clusters[i]
                y, x = dict_['yx']
                r, g, b = dict_['color']
                num_members = dict_['num_members']

                new_clusters[i].number = i
                new_clusters[i].y = y
                new_clusters[i].x = x
                new_clusters[i].r = r
                new_clusters[i].g = g
                new_clusters[i].b = b
                new_clusters[i].num_members = num_members
        except:
            free(new_clusters)
            raise
        if self._c_clusters is not NULL:
            free(self._c_clusters)
        self._c_clusters = new_clusters
        self.num_components = num_new_clusters
        self.initialized = True

    def to_yxmrgb(self):
        cdef cfast_slic.Cluster* cluster
        cdef int i

        cdef np.ndarray[np.int32_t, ndim=2, mode='c'] result = np.ndarray([self.num_components, 6], dtype=np.int32)
        for i in range(0, self.num_components):
            cluster = self._c_clusters + i
            result[i, 0] = cluster.y
            result[i, 1] = cluster.x
            result[i, 2] = cluster.num_members
            result[i, 3] = cluster.r
            result[i, 4] = cluster.g
            result[i, 5] = cluster.b
        return result

    @property
    def clusters(self):
        return self._get_clusters()

    @clusters.setter
    def clusters(self, clusters):
        self._set_clusters(clusters)


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


    cpdef iterate(self, const uint8_t [:, :, ::1] image, int max_iter, float compactness, float min_size_factor, uint8_t quantize_level): 
        if not self.initialized:
            raise RuntimeError("Slic model is not initialized")
        if image.shape[2] != 3:
            raise ValueError("nchan != 3")
        cdef int H = image.shape[0]
        cdef int W = image.shape[1]
        cdef int K = self.num_components
        cdef np.ndarray[np.uint16_t, ndim=2, mode='c'] assignments = np.zeros([H, W], dtype=np.uint16)
        cdef cfast_slic.Cluster* c_clusters = self._c_clusters

        if self._get_name() == 'standard':
            cfast_slic.fast_slic_iterate(
                H,
                W,
                K,
                compactness,
                min_size_factor,
                quantize_level,
                max_iter,
                &image[0, 0, 0],
                c_clusters,
                <uint16_t *>&assignments[0, 0]
            )
        elif self._get_name() == 'avx2':
            cfast_slic.fast_slic_iterate_avx2(
                H,
                W,
                K,
                compactness,
                min_size_factor,
                quantize_level,
                max_iter,
                &image[0, 0, 0],
                c_clusters,
                <uint16_t *>&assignments[0, 0]
            )
        else:
            raise RuntimeError("Not reachable")
        result = assignments.astype(np.int16)
        result[result == 0xFFFF] = -1
        return result

    cpdef get_connectivity(self, const int16_t[:,::1] assignments):
        cdef int H = assignments.shape[0]
        cdef int W = assignments.shape[1]
        cdef int K = self.num_components
        cdef int i, k;
        cdef uint32_t neighbor

        cdef Connectivity* conn;
        with nogil:
            conn = cfast_slic.fast_slic_get_connectivity(H, W, K, <const uint16_t *>&assignments[0, 0])
        return NodeConnectivity.create(conn)

    cpdef get_knn_connectivity(self, const int16_t[:,::1] assignments, size_t num_neighbors):
        cdef int H = assignments.shape[0]
        cdef int W = assignments.shape[1]
        cdef int K = self.num_components
        cdef cfast_slic.Cluster* c_clusters = self._c_clusters
        with nogil:
            conn = cfast_slic.fast_slic_knn_connectivity(H, W, K, c_clusters, num_neighbors)
        return NodeConnectivity.create(conn)

    cpdef get_mask_density(self, const uint8_t[:, ::1] mask, const int16_t[:, ::1] assignments):
        cdef int H = assignments.shape[0]
        cdef int W = assignments.shape[1]
        cdef int K = self.num_components
        cdef const cfast_slic.Cluster* _c_clusters = self._c_clusters

        if mask.shape[0] != H or mask.shape[1] != W:
            raise ValueError("The shape of mask does not match the one of assignments")
        cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] densities = np.ndarray([K], dtype=np.uint8)

        with nogil:
            cfast_slic.fast_slic_get_mask_density(
                H,
                W,
                K,
                _c_clusters,
                <uint16_t *>&assignments[0, 0],
                &mask[0, 0],
                <uint8_t *>&densities[0],
            )
        return densities

    cpdef broadcast_density_to_mask(self, const uint8_t[::1] densities, const int16_t[:, ::1] assignments):
        cdef int H = assignments.shape[0]
        cdef int W = assignments.shape[1]
        cdef int K = self.num_components
        cdef const cfast_slic.Cluster* _c_clusters = self._c_clusters
        if densities.shape[0] != K:
            raise ValueError("The shape of densities should match the number of clusters")
        cdef np.ndarray[np.uint8_t, ndim=2, mode='c'] mask = np.ndarray([H, W], dtype=np.uint8)
        with nogil:
            cfast_slic.fast_slic_cluster_density_to_mask(
                H,
                W,
                K,
                _c_clusters,
                <uint16_t *>&assignments[0, 0],
                &densities[0],
                <uint8_t *>&mask[0, 0],
            )

        return mask

        



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

cdef class NodeConnectivity:
    @staticmethod
    cdef create(Connectivity* conn):
        cdef NodeConnectivity c = NodeConnectivity()
        c._c_connectivity = conn
        return c

    cpdef tolist(self):
        if self._c_connectivity is NULL:
            return []
        result = []
        for k in range(self._c_connectivity.num_nodes):
            k_neighbors = []
            for i in range(self._c_connectivity.num_neighbors[k]):
                neighbor = self._c_connectivity.neighbors[k][i]
                k_neighbors.append(neighbor)
            result.append(k_neighbors)
        return result

    def __dealloc__(self):
        if self._c_connectivity is not NULL:
            cfast_slic.fast_slic_free_connectivity(self._c_connectivity)


def slic_supports_arch(name):
    if name == 'avx2':
        return cfast_slic.fast_slic_supports_avx2() == 1
    return False

