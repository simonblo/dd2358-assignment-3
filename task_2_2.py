import numpy

def jacobi_iteration_numpy(f, n):
    g = numpy.zeros_like(f)
    numpy.add(g[+1:-1, +1:-1], f[+1:-1, +2:  ], g[+1:-1, +1:-1])
    numpy.add(g[+1:-1, +1:-1], f[+1:-1,   :-2], g[+1:-1, +1:-1])
    numpy.add(g[+1:-1, +1:-1], f[+2:  , +1:-1], g[+1:-1, +1:-1])
    numpy.add(g[+1:-1, +1:-1], f[  :-2, +1:-1], g[+1:-1, +1:-1])
    numpy.multiply(g, 0.25, g)
    return g

if __name__ == "__main__":
    n = 1024
    f = numpy.random.random((n, n))
    f[  :+1,   :  ] = 0.0
    f[-1:  ,   :  ] = 0.0
    f[+1:-1,   :+1] = 0.0
    f[+1:-1, -1:  ] = 0.0
    for _ in range(1000):
        f = jacobi_iteration_numpy(f, n)