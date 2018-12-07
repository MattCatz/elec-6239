// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "term.h"

// This is the default block size
const unsigned int BLOCK_SIZE = 32;

__global__ void convolution_basic(matrix_t* image, matrix_t* filter, matrix_t* result) {
    matrix_t sum;
    int rowd,cold,Bx,By,Tx,Ty;
    int i,j,half,start_x,start_y,end_x,end_y;

    Bx = blockIdx.x;
    By = blockIdx.y;
    Tx = threadIdx.x;
    Ty = threadIdx.y;

    sum = 0;

    rowd = By * BLOCK_SIZE + Ty;
    cold = Bx * BLOCK_SIZE + Tx;

    half = (W_SMOOTHING - 1)/2;

    start_y = rowd - half >= 0 ? -half : -rowd;
    end_y = rowd + half < M ? half : M - rowd - 1;
    
    for(i=start_y; i <= end_y; ++i) {
        start_x = cold - half >= 0 ? -half :  - cold;
        end_x = cold + half < M ? half : M - cold - 1;
        for(j=start_x; j <= end_x; ++j) {
            sum += image[M*(rowd+i)+(cold+j)]*filter[W_SMOOTHING*(i+half)+(j+half)];
        }
    }

    result[M*(rowd) + cold] = sum;
}

__global__ void convolution_edge(matrix_t* image, matrix_t* filter, matrix_t* result) {
    matrix_t gx,gy;
    int rowd,cold,Bx,By,Tx,Ty;
    int i,j,half,start_x,start_y,end_x,end_y;

    Bx = blockIdx.x;
    By = blockIdx.y;
    Tx = threadIdx.x;
    Ty = threadIdx.y;

    gx = 0;
    gy = 0;

    rowd = By * BLOCK_SIZE + Ty;
    cold = Bx * BLOCK_SIZE + Tx;

    half = (W_EDGE - 1)/2;

    start_y = rowd - half >= 0 ? -half : -rowd;
    end_y = rowd + half < M ? half : M - rowd - 1;

    for(i=start_y; i <= end_y; ++i) {
        start_x = cold - half >= 0 ? -half :  - cold;
        end_x = cold + half < M ? half : M - cold - 1;
        for(j=start_x; j <= end_x; ++j) {
            gx += image[M*(rowd+i)+(cold+j)]*filter[W_EDGE*(i+half)+(j+half)];
            gy += image[M*(rowd+i)+(cold+j)]*filter[W_EDGE*(j+half)+(i+half)];
        }
    }

    result[M*(rowd) + cold] = sqrt(gx*gx+gy*gy) > 70 ? 255 : 0;
}

int main(void) {

    // Pointers for host
    matrix_t *image;
    matrix_t *gaussian;
    matrix_t *sobel;
    matrix_t *result;

    // Pointers for device memory
    matrix_t *image_d;
    matrix_t *gaussian_d;
    matrix_t *sobel_d;
    matrix_t *result_d;

    // Used to measure performance
    cudaEvent_t start, stop;

    // Used for timing
    float msecTotal = 0.0f;

    printf("Using block size %d\n", BLOCK_SIZE);

    // Allocate host memory for matrices
    // We have to cast our calloc/malloc
    // because cuda is technically a subset
    // of c++ not vanilla c 
    image = (matrix_t *) calloc(M*M, sizeof(matrix_t));
    assert(image != NULL);

    gaussian = (matrix_t *) calloc(W_SMOOTHING*W_SMOOTHING, sizeof(matrix_t));
    assert(gaussian != NULL);

    sobel = (matrix_t *) calloc(W_EDGE*W_EDGE, sizeof(matrix_t));
    assert(sobel != NULL);

    result = (matrix_t *) calloc(M*M, sizeof(matrix_t));
    assert(result != NULL);

    get_image(image);
    generate_guassian_2d(gaussian);
    generate_sobel_2d(sobel);

    save_ppm("Leaves_original_cuda.ppm", image);
    save_g("2d", gaussian);

    // Allocate device memory for matricies
    checkCudaErrors(cudaMalloc((void **) &(image_d), M*M*sizeof(matrix_t)));
    checkCudaErrors(cudaMalloc((void **) &(gaussian_d), W_SMOOTHING*W_SMOOTHING*sizeof(matrix_t)));
    checkCudaErrors(cudaMalloc((void **) &(sobel_d), W_EDGE*W_EDGE*sizeof(matrix_t)));
    checkCudaErrors(cudaMalloc((void **) &(result_d), M*M*sizeof(matrix_t)));

    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(image_d, image, M*M*sizeof(matrix_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gaussian_d, gaussian,  W_SMOOTHING*W_SMOOTHING*sizeof(matrix_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(sobel_d, sobel,  W_EDGE*W_EDGE*sizeof(matrix_t), cudaMemcpyHostToDevice));

    // Allocate CUDA events that we'll use for timing
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Setup execution parameters TODO
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(M / threads.x, M / threads.y);

    printf("Computing result...\n");

    cudaDeviceSynchronize();

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    // Execute the kernel
    convolution_basic <<< grid, threads >>>(image_d, gaussian_d, result_d);
    checkCudaErrors(cudaMemcpy(image_d, result_d, M*M*sizeof(matrix_t), cudaMemcpyDefault));
    convolution_edge <<< grid, threads >>>(image_d, sobel_d, result_d);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(result, result_d, M*M*sizeof(matrix_t), cudaMemcpyDeviceToHost));

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    printf("done in % 3f (sec)\n", msecTotal/1000);

    save_ppm("Leaves_blur_cuda.ppm", result);

    // Clean up memory
    free(image);
    free(gaussian);
    free(sobel);
    checkCudaErrors(cudaFree(image_d));
    checkCudaErrors(cudaFree(gaussian_d));
    checkCudaErrors(cudaFree(sobel_d));

    /* end multiplication */

    return 0;
}
