// System includes
#include <stdio h>
#include <assert h>

// CUDA runtime
#include <cuda_runtime h>

// Helper functions and utilities to work with CUDA
#include <helper_functions h>
#include <helper_cuda h>

#include "term.h"

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

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(matrix_t* image, matrix_t* filter, matrix_t* result) {
    // Data cache: threadIdx x , threadIdx y
    __shared__ float data[ TILE_H * (TILE_W + KERNEL_RADIUS * 2) ];

    // global mem address of this thread
    const int gLoc = threadIdx x +
                            IMUL(blockIdx x, blockDim x) +
                            IMUL(threadIdx y, dataW) +
                            IMUL(blockIdx y, blockDim y) * dataW;


    int x; // image based coordinate

    // original image based coordinate
    const int x0 = threadIdx x + IMUL(blockIdx x, blockDim x);
    const int shift = threadIdx y * (TILE_W + KERNEL_RADIUS * 2);

    // case1: left
    x = x0 - KERNEL_RADIUS;
    if ( x < 0 )
        data[threadIdx x + shift] = 0;
    else
        data[threadIdx x + shift] = d_Data[ gLoc - KERNEL_RADIUS];

    // case2: right
    x = x0 + KERNEL_RADIUS;
    if ( x > dataW-1 )
        data[threadIdx x + blockDim x + shift] = 0;
    else
        data[threadIdx x + blockDim x + shift] = d_Data[gLoc + KERNEL_RADIUS];

    __syncthreads();

    // convolution
    float sum = 0;
    x = KERNEL_RADIUS + threadIdx x;
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
        sum += data[x + i + shift] * d_Kernel[KERNEL_RADIUS + i];

    d_Result[gLoc] = sum;

}

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColGPU(float *d_Result,float *d_Data){
    // Data cache: threadIdx x , threadIdx y
    __shared__ float data[TILE_W * (TILE_H + KERNEL_RADIUS * 2)];

    // global mem address of this thread
    const int gLoc = threadIdx x +
                        IMUL(blockIdx x, blockDim x) +
                        IMUL(threadIdx y, dataW) +
                        IMUL(blockIdx y, blockDim y) * dataW;

    int y; // image based coordinate

    // original image based coordinate
    const int y0 = threadIdx y + IMUL(blockIdx y, blockDim y);
    const int shift = threadIdx y * (TILE_W);

    // case1: upper
    y = y0 - KERNEL_RADIUS;
    if ( y < 0 )
        data[threadIdx x + shift] = 0;
    else
        data[threadIdx x + shift] = d_Data[ gLoc - IMUL(dataW, KERNEL_RADIUS)];

    // case2: lower
    y = y0 + KERNEL_RADIUS;
    const int shift1 = shift + IMUL(blockDim y, TILE_W);
    if ( y > dataH-1 )
        data[threadIdx x + shift1] = 0;
    else
        data[threadIdx x + shift1] = d_Data[gLoc + IMUL(dataW, KERNEL_RADIUS)];

    __syncthreads();

    // convolution
    float sum = 0;
    for (int i = 0; i <= KERNEL_RADIUS*2; i++)
        sum += data[threadIdx x + (threadIdx y + i) * TILE_W] * d_Kernel[i];

    d_Result[gLoc] = sum;

}

int main(void) {
    unsigned int i,j;

    // Pointers for host
    matrix_t *image,
    matrix_t *gaussian;
    matrix_t *sobel_x, *sobel_y;

    // Pointers for device memory
    matrix_t *image_d,
    matrix_t *gaussian_d;
    matrix_t *sobel_x_d, *sobel_y_d;

    // Used to measure performance
    cudaEvent_t start, stop;

    // Used for timing
    float msecTotal = 0 0f;


    printf("Using block size %d\n", BLOCK_SIZE);


    // Allocate host memory for matrices
    // We have to cast our calloc/malloc
    // because cuda is technically a subset
    // of c++ not vanilla c 
    image = (matrix_t *) calloc(M*M, sizeof(matrix_t));
    assert(ah != NULL);

    gaussian = (matrix_t *) calloc(W_SMOOTHING*W_SMOOTHING, sizeof(matrix_t));
    assert(bh != NULL);

    sobel_x = (matrix_t *) calloc(W_EDGE*W_EDGE, sizeof(matrix_t));
    assert(ch != NULL);

    sobel_y = (matrix_t *) calloc(W_EDGE*W_EDGE, sizeof(matrix_t));
    assert(ch != NULL);

    get_image(image);
    generate_guassian(gaussian);
    generate_sobel(sobel_x,sobel_y);

    // Allocate device memory for matricies
    checkCudaErrors(cudaMalloc((void **) &(image_d), M*M*sizeof(matrix_t)));
    checkCudaErrors(cudaMalloc((void **) &(gaussian_d), W_SMOOTHING*W_SMOOTHING*sizeof(matrix_t)));
    checkCudaErrors(cudaMalloc((void **) &(sobel_x_d), W_EDGE*W_EDGE*sizeof(matrix_t)));
    checkCudaErrors(cudaMalloc((void **) &(sobel_y_d), W_EDGE*W_EDGE*sizeof(matrix_t)));

    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(image_d, image, M*M*sizeof(matrix_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gaussian_d, gaussian,  W_SMOOTHING*W_SMOOTHING*sizeof(matrix_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(sobel_x_d, sobel_x,  W_EDGE*W_EDGE*sizeof(matrix_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(sobel_y_d, sobel_y,  W_EDGE*W_EDGE*sizeof(matrix_t), cudaMemcpyHostToDevice));

    // Allocate CUDA events that we'll use for timing
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Setup execution parameters TODO
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(M / threads x, M / threads y);

    printf("Computing result...\n");

    cudaDeviceSynchronize();

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    // Execute the kernel
    convolutionRowGPU <<< grid, threads >>>(cd, ad, bd);
    convolutionColGPU <<< grid, threads >>>(cd, ad, bd);
    convolutionRowGPU <<< grid, threads >>>(cd, ad, bd);
    convolutionColGPU <<< grid, threads >>>(cd, ad, bd);
    MagnatudeGPU <<< grid, threads >>>(cd, ad, bd);


    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(ch, cd, M*M*sizeof(matrix_t), cudaMemcpyDeviceToHost));

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    printf("done in % 3f (sec)\n", msecTotal/1000 0);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(ch, cd, M*M*sizeof(matrix_t), cudaMemcpyDeviceToHost));

    // Clean up memory
    free(image);
    free(gaussian);
    free(sobel_x);
    free(sobel_y);
    checkCudaErrors(cudaFree(image_d));
    checkCudaErrors(cudaFree(gaussian_d));
    checkCudaErrors(cudaFree(sobel_x));
    checkCudaErrors(cudaFree(sobel_y));

    /* end multiplication */

    return 0;
}
