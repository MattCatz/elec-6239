// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

typedef float matrix_t;  

#define INDEX(m, row, col) \
  m[(M) * (row) + (col)]

#define M 4096  //matrix size
#define B 16     //block size
#define W 4     //# of elements for each thread to compute in each dimension.

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

__global__ void matrixMultiply(float *ad, float *bd, float *cd){
    /* declare shared memory arrays for each thread */
    __shared__ float ads[B*W][B*W];
    __shared__ float bds[B*W][B*W];

    /* thread index */
    int tx = threadIdx.x * W;
    int ty = threadIdx.y * W;

    /* calculate row index of a and c element */
    int row = blockIdx.y * B * W + ty;
    /* calculate col index of b and c element */
    int col = blockIdx.x * B * W + tx;

    /* create an array of sums where sum stores the c elements computed by one thread*/
    float sum[W*W];

    for(int i = 0; i < W*W; i++)
        sum[i] = 0.0;


    for(int block = 0; block < M/(B*W); block++){

        /* each thread loads a W*W block into the B*W block */
        for(int i = 0; i < W; i++){
            for(int j = 0; j < W; j++){
                ads[ty+i][tx+j] = ad[(row+i) * M + block * B * W + tx + j];
                bds[ty+i][tx+j] = bd[(block * B * W + ty + i) * M + col + j];
            }
        }
        /* synchronize to make sure blocks are loaded */
        __syncthreads();

        /* each thread computes a W*W block in the B*W block */
        for(int i = 0; i < W; i++){
            for(int j = 0; j < W; j++){
                for(int k = 0; k < B*W; k++)
                    sum[i*W+j] += ads[ty+i][k] * bds[k][tx+j];
            }
        }
        /* synchronize to ensure previous block computation is finished b4 loading new block */
        __syncthreads();
    }

    /* each thread stores a W*W block into the B*W block */
    for(int i = 0; i < W; i++){
        for(int j = 0; j < W; j++)
            cd[(row+i)*M+col+j] = sum[i*W+j];
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


    printf("Using block size %d... with a window of %d...\n", B, W);


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

    dim3 dimGrid(M/(B*W), M/(B*W));
    dim3 dimBlock(B, B);

    /* generate a, b */
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            INDEX(ah,i,j) = ((i+1.0)*(j+1.0)) * (1.0/M);
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

    cudaDeviceSynchronize();

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    /* invoke kernel - perform matrix multiply for c */
    matrixMultiply<<<dimGrid, dimBlock>>>(ad, bd, cd);
    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(ch, cd, M*M*sizeof(matrix_t), cudaMemcpyDeviceToHost));

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    printf("done in %f (sec)\n", msecTotal/1000.0);

    printf("Percent error is %.3f\n", percent_error(ch));

    printf("\nLocal G results:\n");
    matrix_print_some(ch, 2044, 2052, 0, 8);
    printf("\n");

    // Clean up memory
    free(ah);
    free(bh);
    free(ch);
    checkCudaErrors(cudaFree(ad));
    checkCudaErrors(cudaFree(bd));
    checkCudaErrors(cudaFree(cd));
    return 0;
}
