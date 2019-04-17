# cython: language_level=3, boundscheck=False
# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cfast_slic as cs

from cython.operator cimport dereference as deref

cdef extern from "simple-crf.hpp":
    ctypedef int simple_crf_time_t

    cdef cppclass CSimpleCRFFrame "SimpleCRFFrame":
        simple_crf_time_t time
        size_t num_classes
        size_t num_nodes

        void normalize()
        void set_clusters(const cs.Cluster* clusters)
        void set_connectivity(const cs.Connectivity* conn)

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

    @unaries.setter
    def unaries(self, float[:,::1] new_value):
        self._check_demension(new_value)
        self._c_frame.set_unary(&new_value[0, 0])

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

    def __dealloc__(self):
        del self._c_crf


