void ${kernel_name}(float* A, float* B, float* C) {
    const int M = ${M};
    const int N = ${N};
    const int K = ${K};

    const int TILE_M = ${tile_m};
    const int TILE_N = ${tile_n};
    const int TILE_K = ${tile_k};

    #if ${use_openmp}
    #pragma omp parallel for collapse(2)
    #endif
    for (int i = 0; i < M; i += TILE_M) {
        for (int j = 0; j < N; j += TILE_N) {
            // Tile multiplication
            for (int ii = i; ii < min(i + TILE_M, M); ii++) {
                for (int jj = j; jj < min(j + TILE_N, N); jj++) {
                    float sum = 0.0f;
                    #pragma openmp simd
                    for (int kk = k; kk < min(k + TILE_K, K); k++) {
                        sum += A[ii * K + kk] * B[kk * N + jj];
                    }
                    C[ii * N + jj] += sum;
                }
            }
        }
    }
}
