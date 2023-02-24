import h5py
import numpy
import timeit

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
    d = numpy.random.random((n, n))
    d[  :+1,   :  ] = 0.0
    d[-1:  ,   :  ] = 0.0
    d[+1:-1,   :+1] = 0.0
    d[+1:-1, -1:  ] = 0.0
    f = h5py.File("simulation.hdf5", "w")
    f["/simulation/000"] = d
    f["/simulation/000"].attrs["iter"] = 0
    f["/simulation/000"].attrs["size"] = n
    f["/simulation/000"].attrs["time"] = 0
    for i in range(1, 1000):
        a = timeit.default_timer()
        d = jacobi_iteration_numpy(d, n)
        b = timeit.default_timer()
        f["/simulation/" + f"{i:03d}"] = d
        f["/simulation/" + f"{i:03d}"].attrs["iter"] = i
        f["/simulation/" + f"{i:03d}"].attrs["size"] = n
        f["/simulation/" + f"{i:03d}"].attrs["time"] = b - a