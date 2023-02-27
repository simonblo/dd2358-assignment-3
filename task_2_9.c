void jacobi_iteration_c(unsigned int n, double f[n][n], double g[n][n])
{
    for (int i = 1; i != n - 1; ++i)
    {
        for (int j = 1; j != n - 1; ++j)
        {
            g[i][j] = 0.25 * (f[i][j+1] + f[i][j-1] + f[i+1][j] + f[i-1][j]);
        }
    }
}