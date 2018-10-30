// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

typedef float matrix_t;

// This is platform dependent stuff
const int BLOCK_SIZE = 32;

// Size of matrix
// Make sure that matrix is divisable by block_size
const unsigned int M = 4096;

#define INDEX(m, row, col) \
  m[(M) * (row) + (col)]

__global__ void kernel_global(matrix_t *C, matrix_t *A, matrix_t *B) {
    matrix_t sum;
    unsigned int row,col,k;

    sum = 0;

    row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    for (k = 0; k < M; ++k) {
        sum += INDEX(A,row,k) * INDEX(B,k,col);
    }
 
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    INDEX(C,row,col) = sum;
}

__global__ void kernel_local(matrix_t *C, matrix_t *A, matrix_t *B) {
    unsigned int a,b,k,aBegin,aEnd,bBegin,aStep,bStep;
    matrix_t sum;

    // Index of the first sub-matrix of A processed by the block
    aBegin = M * BLOCK_SIZE * blockIdx.y;

    // Index of the last sub-matrix of A processed by the block
    aEnd   = aBegin + M - 1;

    // Step size used to iterate through the sub-matrices of A
    aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    bBegin = BLOCK_SIZE * blockIdx.x;

    // Step size used to iterate through the sub-matrices of B
    bStep  = BLOCK_SIZE * M;

    sum = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ matrix_t As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ matrix_t Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[threadIdx.y][threadIdx.x] = A[a + M * threadIdx.y + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[b + M * threadIdx.y + threadIdx.x];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        #pragma unroll
        for (k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    unsigned int c = M * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
    C[c + M * threadIdx.y + threadIdx.x] = sum;
}

matrix_t percent_error(matrix_t *C) {
    matrix_t total_error, error;
    unsigned int i,j;

    total_error = 0;
    error = 0;

    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            error = fabs(INDEX(C,i,j) - (i+1.0)*(j+1.0)) / ((i+1.0)*(j+1.0));
            total_error += error;
        }
    }
  
  return (total_error * 100) / pow(M,2);
}

int main(void) {
    unsigned int i,j;

    // Pointers for host
    matrix_t *A,*B,*C;

    // Pointers for device memory
    matrix_t *d_A, *d_B, *d_C;

    // Used to measure performance
    cudaEvent_t start, stop;

    // Allocate host memory for matrices
    // We have to cast our calloc/malloc
    // because cuda is technically a subset
    // of c++ not vanilla c.
    A = (matrix_t *) calloc(M*M, sizeof(matrix_t));
    assert(A != NULL);

    B = (matrix_t *) calloc(M*M, sizeof(matrix_t));
    assert(B != NULL);

    C = (matrix_t *) calloc(M*M, sizeof(matrix_t));
    assert(C != NULL);

    // Gen A on host
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            INDEX(A,i,j) = ((i+1.0)*(j+1.0)) * (1.0/M);
        }
    }

    // Gen B on host
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            INDEX(B,i,j) = (j+1.0)/(i+1.0);
        }
    }

    // Allocate device memory for matricies
    checkCudaErrors(cudaMalloc((void **) &(d_A), M*M*sizeof(matrix_t)));
    checkCudaErrors(cudaMalloc((void **) &(d_B), M*M*sizeof(matrix_t)));
    checkCudaErrors(cudaMalloc((void **) &(d_C), M*M*sizeof(matrix_t)));

    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(d_A, A, M*M*sizeof(matrix_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, M*M*sizeof(matrix_t), cudaMemcpyHostToDevice));

    // Allocate CUDA events that we'll use for timing
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Setup execution parameters TODO
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(M / threads.x, M / threads.y);

    printf("Computing result using CUDA Kernel...\n");

    cudaDeviceSynchronize();

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    // Execute the kernel
    kernel_local <<< grid, threads >>>(d_C, d_A, d_B);
    // Execute the kernel
    kernel_global <<< grid, threads >>>(d_C, d_A, d_B);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(C, d_C, M*M*sizeof(matrix_t), cudaMemcpyDeviceToHost));

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Mit for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    printf("done in %.3f (sec)\n", msecTotal/1000.0);

    printf("Percent error is %.3f\n", percent_error(C));

    // Clean up memory
    free(A);
    free(B);
    free(C);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    /* end multiplication */

    return 0;
}
