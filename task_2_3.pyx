import cython
import numpy

cimport numpy

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def jacobi_iteration_cython(double[:,:] f, unsigned int n):
    cdef double[:,:] g = numpy.zeros((n, n), numpy.double)
    cdef unsigned int i
    cdef unsigned int j
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            g[i][j] = 0.25 * (f[i][j+1] + f[i][j-1] + f[i+1][j] + f[i-1][j])
    return g