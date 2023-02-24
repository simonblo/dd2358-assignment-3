import cupy

def jacobi_iteration_cupy(f, n):
    g = cupy.zeros_like(f)
    cupy.add(g[+1:-1, +1:-1], f[+1:-1, +2:  ], g[+1:-1, +1:-1])
    cupy.add(g[+1:-1, +1:-1], f[+1:-1,   :-2], g[+1:-1, +1:-1])
    cupy.add(g[+1:-1, +1:-1], f[+2:  , +1:-1], g[+1:-1, +1:-1])
    cupy.add(g[+1:-1, +1:-1], f[  :-2, +1:-1], g[+1:-1, +1:-1])
    cupy.multiply(g, 0.25, g)
    cupy.cuda.Stream.null.synchronize()
    return g

if __name__ == "__main__":
    n = 1024
    f = cupy.random.random((n, n))
    f[  :+1,   :  ] = 0.0
    f[-1:  ,   :  ] = 0.0
    f[+1:-1,   :+1] = 0.0
    f[+1:-1, -1:  ] = 0.0
    for _ in range(1000):
        f = jacobi_iteration_cupy(f, n)