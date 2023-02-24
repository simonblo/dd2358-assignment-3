import cupy
import matplotlib.pyplot as plt
import numpy
import pytest
import task_2_3
import timeit
import torch

def test_jacobi_iteration_numpy():
    n = 128
    f = init_grid_list(n)
    g = numpy.array(f)
    for _ in range(1000):
        f = jacobi_iteration_list(f, n)
        g = jacobi_iteration_numpy(g, n)
        assert numpy.allclose(f, g)

def test_jacobi_iteration_cython():
    n = 128
    f = init_grid_list(n)
    g = numpy.array(f)
    for _ in range(1000):
        f = jacobi_iteration_list(f, n)
        g = jacobi_iteration_cython(g, n)
        assert numpy.allclose(f, g)

def test_jacobi_iteration_torch():
    n = 128
    f = init_grid_list(n)
    g = torch.tensor(f).cuda()
    for _ in range(1000):
        f = jacobi_iteration_list(f, n)
        g = jacobi_iteration_torch(g, n)
        assert torch.allclose(torch.tensor(f).cuda(), g)

def test_jacobi_iteration_cupy():
    n = 128
    f = init_grid_list(n)
    g = cupy.array(f)
    for _ in range(1000):
        f = jacobi_iteration_list(f, n)
        g = jacobi_iteration_cupy(g, n)
        assert cupy.allclose(f, g)

def init_grid_list(n):
    return init_grid_numpy(n).tolist()

def init_grid_numpy(n):
    return numpy.pad(numpy.random.random((n-2, n-2)), 1)

def init_grid_cython(n):
    return init_grid_numpy(n)

def init_grid_torch(n):
    return torch.tensor(init_grid_list(n)).cuda()

def init_grid_cupy(n):
    return cupy.array(init_grid_list(n))

def jacobi_iteration_list(f, n):
    g = [[0.0] * n for _ in range(n)]
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            g[i][j] = 0.25 * (f[i][j+1] + f[i][j-1] + f[i+1][j] + f[i-1][j])
    return g

def jacobi_iteration_numpy(f, n):
    g = numpy.zeros((n, n))
    numpy.add(g[+1:-1, +1:-1], f[+1:-1, +2:  ], g[+1:-1, +1:-1])
    numpy.add(g[+1:-1, +1:-1], f[+1:-1,   :-2], g[+1:-1, +1:-1])
    numpy.add(g[+1:-1, +1:-1], f[+2:  , +1:-1], g[+1:-1, +1:-1])
    numpy.add(g[+1:-1, +1:-1], f[  :-2, +1:-1], g[+1:-1, +1:-1])
    numpy.multiply(g, 0.25, g)
    return g

def jacobi_iteration_cython(f, n):
    return task_2_3.jacobi_iteration_cython(f, n)

def jacobi_iteration_torch(f, n):
    g = torch.zeros_like(f)
    torch.add(g[+1:-1, +1:-1], f[+1:-1, +2:  ], alpha=0.25, out=g[+1:-1, +1:-1])
    torch.add(g[+1:-1, +1:-1], f[+1:-1,   :-2], alpha=0.25, out=g[+1:-1, +1:-1])
    torch.add(g[+1:-1, +1:-1], f[+2:  , +1:-1], alpha=0.25, out=g[+1:-1, +1:-1])
    torch.add(g[+1:-1, +1:-1], f[  :-2, +1:-1], alpha=0.25, out=g[+1:-1, +1:-1])
    torch.cuda.synchronize()
    return g

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
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    fig.supxlabel("Grid size (N×N)")
    fig.supylabel("Iteration time (s)")
    xticks = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    yticks = [5e-6, 5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 5e-0]
    f = []
    f.append((init_grid_list,   jacobi_iteration_list,   [], [], "List"))
    f.append((init_grid_numpy,  jacobi_iteration_numpy,  [], [], "Numpy"))
    f.append((init_grid_cython, jacobi_iteration_cython, [], [], "Cython"))
    f.append((init_grid_torch,  jacobi_iteration_torch,  [], [], "Torch"))
    f.append((init_grid_cupy,   jacobi_iteration_cupy,   [], [], "Cupy"))
    for n in xticks:
        k = len(f)
        for i in range(k):
            g = f[i][0](n)
            for j in range(10):
                print(f"\r{n:4d}: {(i+1):d}/{k:d} {(10*j):d}%", end="")
                a = timeit.default_timer()
                g = f[i][1](g, n)
                b = timeit.default_timer()
                f[i][2].append(n)
                f[i][3].append(b - a)
        print(f"\r{n:4d}: {k:d}/{k:d} 100%")
    for v in f:
        ax.scatter(v[2], v[3], label=v[4], alpha=0.25)
    ax.grid()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(xticks[0], xticks[-1])
    plt.ylim(yticks[0], yticks[-1])
    plt.xticks(xticks, [f"{v:d}×{v:d}" for v in xticks])
    plt.yticks(yticks, [f"{v:.6f}"     for v in yticks])
    plt.minorticks_off()
    plt.show()