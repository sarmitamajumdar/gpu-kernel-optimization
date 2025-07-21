#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define TILE_SIZE 16
#define N 512

// CUDA kernel using shared memory
__global__ void MatMulShared(float *A, float *B, float *C, int n) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int tileIdx = 0; tileIdx < (n / TILE_SIZE); ++tileIdx) {
        if (row < n && tileIdx * TILE_SIZE + threadIdx.x < n)
            Asub[threadIdx.y][threadIdx.x] = A[row * n + tileIdx * TILE_SIZE + threadIdx.x];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        if (tileIdx * TILE_SIZE + threadIdx.y < n && col < n)
            Bsub[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE_SIZE + threadIdx.y) * n + col];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(N / TILE_SIZE, N / TILE_SIZE);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

    MatMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("C[0][0] = %.2f\n", h_C[0]);
    printf("C[100][100] = %.2f\n", h_C[100 * N + 100]);
    printf("C[511][511] = %.2f\n", h_C[511 * N + 511]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
