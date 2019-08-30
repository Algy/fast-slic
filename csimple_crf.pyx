# cython: language_level=3, boundscheck=False
# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cfast_slic as cs

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.stdint cimport uint32_t, int32_t
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement

cdef extern from "src/simple-crf.hpp":
    ctypedef int simple_crf_time_t

    ctypedef struct SimpleCRFParams:
        float spatial_w
        float temporal_w
        float spatial_srgb
        float temporal_srgb
        float spatial_sxy
        float spatial_smooth_w
        float spatial_smooth_sxy

    cdef cppclass CSimpleCRFFrame "SimpleCRFFrame":
        simple_crf_time_t time
        size_t num_classes
        size_t num_nodes

        void normalize()
        void set_clusters(const cs.Cluster* clusters)
        void get_clusters(cs.Cluster* clusters)
        void set_connectivity(const cs.Connectivity* conn)
        const vector[int]& connected_nodes(int node) except +

        void get_unary(float *unaries_out) const
        void set_unbiased()
        void set_mask(const int* classes, float confidence)
        void set_proba(const float* probas);
        void set_unary(const float* unaries);

        void get_inferred(float *out) const
        void reset_inferred() nogil

        size_t space_size()
        float calc_temporal_pairwise_energy(int node, const CSimpleCRFFrame& other) nogil
        float calc_spatial_pairwise_energy(int node_i, int node_j) nogil

    cdef cppclass CSimpleCRF "SimpleCRF":
        SimpleCRFParams params

        CSimpleCRF(size_t num_classes, size_t num_nodes) except +
        simple_crf_time_t get_first_time()
        simple_crf_time_t get_last_time()
        size_t get_num_frames()
        simple_crf_time_t pop_frame()
        CSimpleCRFFrame& get_frame(simple_crf_time_t time) except +
        CSimpleCRFFrame& push_frame() except +
        size_t space_size()
        void initialize() nogil
        void inference(size_t max_iter) nogil



cdef class SimpleCRFFrame:
    cdef CSimpleCRFFrame* _c_frame
    cdef readonly object parent_crf

    def __cinit__(self, parent_crf):
        self.parent_crf = parent_crf # Prevent unintended GC: because CRF object owns Frame, associated frame object would become dangling pointer if crf python object got collected. 


    @property
    def time(self):
        return self._c_frame.time

    @property
    def space_size(self):
        return self._c_frame.space_size()

    @property
    def unaries(self):
        cdef np.ndarray[np.float32_t,  ndim=2, mode='c'] unaries = self._fresh_buffer()
        self._c_frame.get_unary(&unaries[0, 0])
        return unaries

    def get_yxmrgb(self):
        result = []
        cdef cs.Cluster* clusters = <cs.Cluster*>malloc(sizeof(cs.Cluster) * self.num_nodes)
        try:
            self._c_frame.get_clusters(clusters)
            for i in range(self.num_nodes):
                result.append([
                    clusters[i].y,
                    clusters[i].x,
                    clusters[i].num_members,
                    clusters[i].r,
                    clusters[i].g,
                    clusters[i].b,
                ])
        finally:
            free(clusters)
        return result

    cpdef set_yxmrgb(self, int32_t[:,::1] yxmrgb):
        if self.num_nodes != yxmrgb.shape[0]:
            raise ValueError("Expected the first dimension of yxmrgb to equal to {}".format(self.num_nodes))
        if 6 != yxmrgb.shape[1]:
            raise ValueError("Expected the second dimension of yxmrgb to equal to 6")
        cdef int i, y, x, r, g, b
        cdef cs.Cluster *clusters = <cs.Cluster*>malloc(sizeof(cs.Cluster) * len(yxmrgb));
        try:
            for i in range(yxmrgb.shape[0]):
                clusters[i].y = yxmrgb[i, 0]
                clusters[i].x = yxmrgb[i, 1]
                clusters[i].num_members = yxmrgb[i, 2]
                clusters[i].r = yxmrgb[i, 3]
                clusters[i].g = yxmrgb[i, 4]
                clusters[i].b = yxmrgb[i, 5]
                clusters[i].number = i
            self._c_frame.set_clusters(clusters)
        finally:
            free(clusters)

    def get_connectivity(self):
        cdef size_t i
        cdef const vector[int]* nodes
        result = []
        for node in range(self.num_nodes):
            nodes = &self._c_frame.connected_nodes(node)
            result.append(deref(nodes))
        return result
    
    cpdef _set_slic_connectivity(self, cs.NodeConnectivity connectivity):
        self._c_frame.set_connectivity(connectivity._c_connectivity)

    def set_connectivity(self, connectivity):
        cdef cs.Connectivity c_conn
        cdef int i, k, num_neighbor
        cdef uint32_t neighbor
        cdef cs.Connectivity* c_connectivity
        if isinstance(connectivity, cs.NodeConnectivity):
            self._set_slic_connectivity(<cs.NodeConnectivity>connectivity)
            return 

        if len(connectivity) != self.num_nodes:
            raise ValueError("Expected len(connectivity) to be {}".format(self.num_nodes))

        c_conn.num_nodes = len(connectivity)
        c_conn.num_neighbors = NULL
        c_conn.neighbors = NULL
        try:
            c_conn.num_neighbors = <int*>malloc(sizeof(int) * len(connectivity))
            c_conn.neighbors = <uint32_t**>malloc(sizeof(uint32_t *) * len(connectivity))
            memset(c_conn.neighbors, 0, sizeof(uint32_t *) * len(connectivity))

            for i, neighbors in enumerate(connectivity):
                c_conn.num_neighbors[i] = len(neighbors)
                c_conn.neighbors[i] = <uint32_t*>malloc(sizeof(uint32_t) * len(neighbors))
                for k, neighbor in enumerate(neighbors):
                    c_conn.neighbors[i][k] = neighbor
            self._c_frame.set_connectivity(&c_conn)
        finally:
            if c_conn.neighbors is not NULL:
                for i in range(len(connectivity)):
                    if c_conn.neighbors[i] is not NULL:
                        free(c_conn.neighbors[i])
                free(c_conn.neighbors)

            if c_conn.num_neighbors is not NULL:
                free(c_conn.num_neighbors)



    @unaries.setter
    def unaries(self, float[:,::1] new_value):
        self._check_demension(new_value)
        self._c_frame.set_unary(&new_value[0, 0])

    @property
    def num_nodes(self):
        return self._c_frame.num_nodes

    @property
    def num_classes(self):
        return self._c_frame.num_classes

    def set_unbiased(self):
        self._c_frame.set_unbiased()

    def set_mask(self, int[::1] classes, float confidence):
        if <size_t>classes.shape[0] != self._c_frame.num_nodes:
            raise ValueError("The dimension of class array should match the number of nodes {}".format(self._c_frame.num_nodes))
        self._c_frame.set_mask(&classes[0], confidence)

    def set_proba(self, float[:, ::1] proba):
        self._check_demension(proba)
        self._c_frame.set_proba(&proba[0, 0])

    def get_inferred(self):
        cdef np.ndarray[np.float32_t,  ndim=2, mode='c'] proba = self._fresh_buffer()
        self._c_frame.get_inferred(&proba[0, 0])
        return proba

    def reset_inferred(self):
        self._c_frame.reset_inferred()

    def temporal_pairwise_energy(self, int node_i, SimpleCRFFrame other):
        cdef float result
        if not isinstance(other, SimpleCRFFrame):
            raise TypeError("not a crf frame")
        cdef CSimpleCRFFrame *other_frame = other._c_frame
        if <size_t>node_i >= self._c_frame.num_nodes:
            raise ValueError("node number is out of range")

        with nogil:
            result = self._c_frame.calc_temporal_pairwise_energy(node_i, deref(other_frame))
        return result

    def spatial_pairwise_energy(self, int node_i, int node_j):
        cdef float result
        if <size_t>node_i >= self._c_frame.num_nodes or <size_t>node_j >= self._c_frame.num_nodes:
            raise ValueError("node number is out of range")
        with nogil:
            result = self._c_frame.calc_spatial_pairwise_energy(node_i, node_j)
        return result

    cdef _check_demension(self, float[:, ::1] arr):
        if <size_t>arr.shape[0] != self._c_frame.num_classes:
            raise ValueError("The first dimension of array should match the number of classes {}".format(self._c_frame.num_classes))
        if <size_t>arr.shape[1] != self._c_frame.num_nodes:
            raise ValueError("The second dimension of array should match the number of nodes {}".format(self._c_frame.num_nodes))

    cdef np.ndarray[np.float32_t,  ndim=2, mode='c'] _fresh_buffer(self):
        return np.zeros(
            [self._c_frame.num_classes, self._c_frame.num_nodes],
            dtype=np.float32
        )



cdef class SimpleCRF:
    cdef CSimpleCRF* _c_crf
    def __cinit__(self, size_t num_classes, size_t num_nodes):
        self._c_crf = new CSimpleCRF(num_classes, num_nodes)

    @property
    def spatial_w(self):
        return self._c_crf.params.spatial_w

    @spatial_w.setter
    def spatial_w(self, float spatial_w):
        self._c_crf.params.spatial_w = spatial_w

    @property
    def spatial_srgb(self):
        return self._c_crf.params.spatial_srgb

    @spatial_srgb.setter
    def spatial_srgb(self, float spatial_srgb):
        self._c_crf.params.spatial_srgb = spatial_srgb

    @property
    def spatial_sxy(self):
        return self._c_crf.params.spatial_sxy

    @spatial_sxy.setter
    def spatial_sxy(self, float spatial_sxy):
        self._c_crf.params.spatial_sxy = spatial_sxy

    @property
    def temporal_w(self):
        return self._c_crf.params.temporal_w

    @temporal_w.setter
    def temporal_w(self, float temporal_w):
        self._c_crf.params.temporal_w = temporal_w

    @property
    def temporal_srgb(self):
        return self._c_crf.params.temporal_srgb

    @temporal_srgb.setter
    def temporal_srgb(self, float temporal_srgb):
        self._c_crf.params.temporal_srgb = temporal_srgb

    @property
    def spatial_smooth_w(self):
        return self._c_crf.params.spatial_smooth_w

    @spatial_smooth_w.setter
    def spatial_smooth_w(self, float spatial_smooth_w):
        self._c_crf.params.spatial_smooth_w = spatial_smooth_w

    @property
    def spatial_smooth_sxy(self):
        return self._c_crf.params.spatial_smooth_sxy

    @spatial_smooth_sxy.setter
    def spatial_smooth_sxy(self, float spatial_smooth_sxy):
        self._c_crf.params.spatial_smooth_sxy = spatial_smooth_sxy

    @property
    def first_time(self):
        return self._c_crf.get_first_time()

    @property
    def last_time(self):
        return self._c_crf.get_last_time()

    @property
    def num_frames(self):
        return self._c_crf.get_num_frames()

    @property
    def space_size(self):
        return self._c_crf.space_size()

    def get_frame(self, simple_crf_time_t time):
        cdef CSimpleCRFFrame* frame = &self._c_crf.get_frame(time)
        cdef SimpleCRFFrame result = SimpleCRFFrame(self)
        result._c_frame = frame
        return result

    cpdef push_slic_frame(self, slic, knn=None):
        frame = self.push_frame()
        frame.set_yxmrgb(slic.slic_model.to_yxmrgb())
        if knn is None:
            frame.set_connectivity(slic.slic_model.get_connectivity(slic.last_assignment))
        else:
            frame.set_connectivity(slic.slic_model.get_knn_connectivity(slic.last_assignment, knn))
        frame.set_unbiased()
        return frame

    cpdef push_frame(self):
        cdef CSimpleCRFFrame* frame = &self._c_crf.push_frame()
        cdef SimpleCRFFrame result = SimpleCRFFrame(self)
        result._c_frame = frame
        return result
      
    cpdef pop_frame(self):
        return self._c_crf.pop_frame()

    def initialize(self):
        with nogil:
            self._c_crf.initialize()

    def inference(self, size_t max_iter):
        with nogil:
            self._c_crf.inference(max_iter)

    def __dealloc__(self):
        del self._c_crf


