import torch

def jacobi_iteration_torch(f, n):
    g = torch.zeros_like(f)
    torch.add(g[+1:-1, +1:-1], f[+1:-1, +2:  ], alpha=0.25, out=g[+1:-1, +1:-1])
    torch.add(g[+1:-1, +1:-1], f[+1:-1,   :-2], alpha=0.25, out=g[+1:-1, +1:-1])
    torch.add(g[+1:-1, +1:-1], f[+2:  , +1:-1], alpha=0.25, out=g[+1:-1, +1:-1])
    torch.add(g[+1:-1, +1:-1], f[  :-2, +1:-1], alpha=0.25, out=g[+1:-1, +1:-1])
    torch.cuda.synchronize()
    return g

if __name__ == "__main__":
    n = 1024
    f = torch.rand((n, n)).cuda()
    f[  :+1,   :  ] = 0.0
    f[-1:  ,   :  ] = 0.0
    f[+1:-1,   :+1] = 0.0
    f[+1:-1, -1:  ] = 0.0
    for _ in range(1000):
        f = jacobi_iteration_torch(f, n)