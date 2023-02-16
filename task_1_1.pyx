import cython
import numpy
import timeit

cimport numpy

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def measure_bandwidth_cython(unsigned int n):
    cdef unsigned int i
    cdef double t0, t1, t2, t3, t4
    cdef double r0, r1, r2, r3
    cdef float[:] a = numpy.full(n, 1.0, dtype=numpy.float32)
    cdef float[:] b = numpy.full(n, 2.0, dtype=numpy.float32)
    cdef float[:] c = numpy.full(n, 0.0, dtype=numpy.float32)
    t0 = timeit.default_timer()
    for i in range(n):
        c[i] = a[i]
    t1 = timeit.default_timer()
    for i in range(n):
        b[i] = 2.0 * c[i]
    t2 = timeit.default_timer()
    for i in range(n):
        c[i] = a[i] * b[i]
    t3 = timeit.default_timer()
    for i in range(n):
        a[i] = b[i] + 2.0 * c[i]
    t4 = timeit.default_timer()
    r0 = ((2 * 8 * n) / (t1 - t0)) * 1e-6
    r1 = ((3 * 8 * n) / (t2 - t1)) * 1e-6
    r2 = ((2 * 8 * n) / (t3 - t2)) * 1e-6
    r3 = ((3 * 8 * n) / (t4 - t3)) * 1e-6
    return [r0, r1, r2, r3]