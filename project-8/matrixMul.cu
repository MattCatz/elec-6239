// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

typedef float matrix_t;

// This is the default block size
const unsigned int BLOCK_SIZE = 16;

// Size of matrix
// Make sure that matrix is divisable by block_size
const unsigned int M = 4096;

// Size of shared memory array
const unsigned int Mds = BLOCK_SIZE;

#define INDEX(m, row, col) \
  m[(M) * (row) + (col)]

__global__ void kernel_global(matrix_t *C, matrix_t *A, matrix_t *B) {
    matrix_t sum;
    unsigned int rowd,cold,k,Bx,By,Tx,Ty;

    Bx = blockIdx.x;
    By = blockIdx.y;
    Tx = threadIdx.x;
    Ty = threadIdx.y;

    sum = 0;

    rowd = By * BLOCK_SIZE + Ty;
    cold = Bx * BLOCK_SIZE + Tx;

    for (k = 0; k < M; ++k) {
        sum += INDEX(A,rowd,k) * INDEX(B,k,cold);
    }
 
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    INDEX(C,rowd,cold) = sum;
}

__global__ void kernel_local(matrix_t *C, matrix_t *A, matrix_t *B) {
    unsigned int a,b,k,start_a,end,start_b,start_c,offset_a,offset_b;
    unsigned int Bx,By,Tx,Ty;
    unsigned int rowd,cold,rowds,colds;
    matrix_t sum;

    __shared__ matrix_t ads[Mds][Mds];
    __shared__ matrix_t bds[Mds][Mds];

    Bx = blockIdx.x;
    By = blockIdx.y;
    Tx = threadIdx.x;
    Ty = threadIdx.y;

    // Start of the blocks
    start_a = M * BLOCK_SIZE * By;
    start_b = BLOCK_SIZE * Bx;
    start_c = M * BLOCK_SIZE * By + BLOCK_SIZE * Bx;

    // Offset between loop iterations
    offset_a = BLOCK_SIZE;
    offset_b = BLOCK_SIZE * M;

    // End of the blocks
    end = start_a + M - 1;

    sum = 0;

    rowd = M * Ty;
    cold = Tx + start_c;

    rowds = Ty;
    colds = Tx;

    for (a = start_a, b = start_b; a <= end; a += offset_a, b += offset_b) {

        // Load into shared memory
        ads[rowds][colds] = A[a + M * Ty + Tx];
        bds[rowds][colds] = B[b + M * Ty + Tx];

        // Wait for all threads to get here
        __syncthreads();

        for (k = 0; k < BLOCK_SIZE; ++k) {
            sum += ads[Ty][k] * bds[k][Tx];
        }

        // Wait for all threads to get here
        __syncthreads();
    }

    // Write back to global memory
    C[rowd + cold] = sum;
}

float percent_error(matrix_t *C) {
    float total_error, error;
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

void matrix_print_some(matrix_t *m, const size_t X1, const size_t X2, const size_t Y1, const size_t Y2) {
   int i,j;
   for (i = X1; i <= X2; i++) {
      for (j = Y1; j < Y2; j++) {
         printf("%.3f, ", INDEX(m,i,j)); // Notice only 3 digits
      }
      printf("\n");
   }
}

int main(void) {
    unsigned int i,j;

    // Pointers for host
    matrix_t *ah,*bh,*ch;

    // Pointers for device memory
    matrix_t *ad, *bd, *cd;

    // Used to measure performance
    cudaEvent_t start, stop;

    // Used for timing
    float msecTotal = 0.0f;


    printf("Using block size %d...\n", BLOCK_SIZE);


    // Allocate host memory for matrices
    // We have to cast our calloc/malloc
    // because cuda is technically a subset
    // of c++ not vanilla c.
    ah = (matrix_t *) calloc(M*M, sizeof(matrix_t));
    assert(ah != NULL);

    bh = (matrix_t *) calloc(M*M, sizeof(matrix_t));
    assert(bh != NULL);

    ch = (matrix_t *) calloc(M*M, sizeof(matrix_t));
    assert(ch != NULL);

    // Gen A on host
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            INDEX(ah,i,j) = ((i+1.0)*(j+1.0)) * (1.0/M);
        }
    }

    // Gen B on host
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            INDEX(bh,i,j) = (j+1.0)/(i+1.0);
        }
    }

    // Allocate device memory for matricies
    checkCudaErrors(cudaMalloc((void **) &(ad), M*M*sizeof(matrix_t)));
    checkCudaErrors(cudaMalloc((void **) &(bd), M*M*sizeof(matrix_t)));
    checkCudaErrors(cudaMalloc((void **) &(cd), M*M*sizeof(matrix_t)));

    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(ad, ah, M*M*sizeof(matrix_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(bd, bh, M*M*sizeof(matrix_t), cudaMemcpyHostToDevice));

    // Allocate CUDA events that we'll use for timing
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Setup execution parameters TODO
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(M / threads.x, M / threads.y);

    printf("Computing result using CUDA Kernel and local memory...\n");

    cudaDeviceSynchronize();

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    // Execute the kernel
    kernel_local <<< grid, threads >>>(cd, ad, bd);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(ch, cd, M*M*sizeof(matrix_t), cudaMemcpyDeviceToHost));

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    printf("done in %.3f (sec)\n", msecTotal/1000.0);

    printf("Percent error is %.3f\n", percent_error(ch));

    printf("\nLocal G results:\n");
    matrix_print_some(ch, 2044, 2052, 0, 8);
    printf("\n");

    printf("Computing result using CUDA Kernel and global memory...\n");

    cudaDeviceSynchronize();

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    // Execute the kernel
    kernel_global <<< grid, threads >>>(cd, ad, bd);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(ch, cd, M*M*sizeof(matrix_t), cudaMemcpyDeviceToHost));

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    printf("done in %.3f (sec)\n", msecTotal/1000.0);

    printf("Percent error is %.3f\n", percent_error(ch));

    printf("\nGlobal G results:\n");
    matrix_print_some(ch, 2044, 2052, 0, 8);
    printf("\n");

    // Clean up memory
    free(ah);
    free(bh);
    free(ch);
    checkCudaErrors(cudaFree(ad));
    checkCudaErrors(cudaFree(bd));
    checkCudaErrors(cudaFree(cd));

    /* end multiplication */

    return 0;
}
