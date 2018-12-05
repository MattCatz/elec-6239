#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <math.h>

#include "term.h"

int world_size, world_rank;

void send_partitions(matrix_t *image, int window, int chunk, int half) {
	int dest,offset;
    // Start at 1 because I dont have to send to myself
	for (dest = 1; dest < world_size; dest++) {
      // We subtract half to get the stuff above us
		offset = (dest * chunk - half)*M;

		if (offset + (window*M) > M*M) {
        // Make sure that we do not send anything past our image
			window = window - half;
		}

		MPI_Send(&image[offset], window*M, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
	}
}

void collect_partitions(matrix_t *image, int window, int chunk) {
	int j,offset,source;
	for (j = 1; j < world_size; j++) {
		source = j;
		offset = (source * chunk)*M;
		MPI_Recv(&image[offset], chunk*M, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}

void master_node() {}

void salve_node() {}

int main(int argc, char** argv) {
    // Initialize the MPI environment
	MPI_Init(NULL, NULL);

    // Get the number of processes
	// int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
	// int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    /* Program begins */

	int half,edge_half,chunk;
	int i,j,k;
	int source,dest;
	int window,offset;
	double start,end;


    // Communication buffers
	matrix_t *buffer,*edge;
    // Filter and pointer used for looping
	matrix_t *filter,*smoothed,*hx,*hy;

	half = W_SMOOTHING/2;

	edge_half = W_EDGE/2;


    /* chunk is the number of rows each thread is responsible for */
	chunk = M / world_size;

    /* since we need information above and below the chunk we call that our
       window of rows we need to perform convolution on out chunk */
	window = chunk + (W_SMOOTHING-1);

    // We subtract half to get the stuff above us
    // offset = (world_rank * chunk - half)*M;

	filter = calloc(W_SMOOTHING, sizeof(matrix_t));
	assert(filter != NULL);

	hx = calloc(W_EDGE*W_EDGE, sizeof(matrix_t));
	assert(hx != NULL);

	hy = calloc(W_EDGE*W_EDGE, sizeof(matrix_t));
	assert(hy != NULL);

	smoothed = calloc(chunk*M, sizeof(matrix_t));
	assert(smoothed != NULL);

    // Generating filter
	generate_guassian(filter);
	generate_sobel(hx,hy);

	if (world_rank == 0) {
		matrix_t *image;

		image = calloc(M*M, sizeof(matrix_t));
		assert(image != NULL);

        // Generating image
		get_image(image);
		save_ppm("Original.ppm", image);

		start = MPI_Wtime();

		// Send out the original image
		send_partitions(image, window, chunk, half);

		for(i=0; i < chunk; ++i) {
			matrix_t temp[M];

			/* Do the row wise de-noise convolution */
			for(j=0; j < M; ++j) {
				matrix_t sum = 0;
				int start_y = i - half >= 0 ? -half : -i;
				int end_y = M - i > half ? half : (M - i - 1);
				for (k = start_y; k <= end_y; ++k) {
					sum += image[(M*(i+k))+j]*filter[-k+half];
				}
				temp[j] = sum;
			}

			/* Do the column wise de-noise convolution */
			for(j=0; j < M; ++j) {
				matrix_t sum = 0;
				int start_x = j - half >= 0 ? -half :  -j;
				int end_x = M - j > half ? half : (M - j - 1);
				for (k = start_x; k <= end_x; ++k) {
					sum += temp[j+k]*filter[-k+half];
				}
				smoothed[(M*i) + j] = sum;
			}
		}

		memcpy(image,smoothed,chunk*M*sizeof(matrix_t));

		collect_partitions(image, window, chunk);

		// TODO use the correct half to shrink memory
		send_partitions(image, window, chunk, half);

		for(i=0; i < chunk; ++i) {
			matrix_t gx[M],gy[M];
			for(j=0; j < M; ++j) {
				gx[j] = 0;
				gy[j] = 0;
				int start_y = i - edge_half >= 0 ? -edge_half : -i;
				int end_y = M - i > edge_half ? edge_half : (M - i - 1);
				for (k = start_y; k <= end_y; ++k) {
					gy[j] += image[(M*(i+k))+j]*hy[-k+edge_half];
					gx[j] += image[(M*(i+k))+j]*hx[-k+edge_half];
				}
			}

			for(j=0; j < M; ++j) {
				matrix_t sum_x = 0;
				matrix_t sum_y = 0;
				int start_x = j - edge_half >= 0 ? -edge_half :  -j;
				int end_x = M - j > edge_half ? edge_half : (M - j - 1);
				for (k = start_x; k <= end_x; ++k) {
					sum_x += gx[j+k]*hy[-k+edge_half];
					sum_y += gy[j+k]*hx[-k+edge_half];
				}
				smoothed[(M*i) + j] = sqrtf(sum_x*sum_x+sum_y*sum_y);
			}
		}

		memcpy(image,smoothed,chunk*M*sizeof(matrix_t));

		collect_partitions(image, window, chunk);

		end = MPI_Wtime();

		printf("\nTotal time with %d threads: %f\n\n",world_size,end-start);
		save_ppm("Leaves_blur.ppm", image);

      } else { // Begin not master node stuff

      	buffer = calloc(window*M, sizeof(matrix_t));
      	assert(buffer != NULL);

        source = 0; // Master node
        MPI_Recv(buffer, window*M, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for(i=0; i < chunk; ++i) {
			// printf("%i\n", i);
			matrix_t temp[M];
			for(j=0; j < M; ++j) {
				matrix_t sum = 0;
				int start_y = 1 ? -half : -i;
				int end_y = i + half < window - half ? half : (chunk - i - 1);
				for (k = start_y; k <= end_y; ++k) {
					// if (i >= chunk - 1 && j < 10) printf("%d %d %d\n", k, start_y,end_y);
					sum += buffer[(M*(i-k+half))+j]*filter[half-k];
				}
				// smoothed[(M*i) + j] = sum;
				temp[j] = sum;
			}

			for(j=0; j < M; ++j) {
				matrix_t sum = 0;
				int start_x = j - half >= 0 ? -half :  -j;
				int end_x = M - j > half ? half : (M - j - 1);
				for (k = start_x; k <= end_x; ++k) {
					sum += temp[j+k]*filter[half-k];
				}
				smoothed[(M*i) + j] = sum;
			}
		}

		// Send back the smoothed window
        dest = 0;
		MPI_Send(smoothed, chunk*M, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
		MPI_Recv(buffer, window*M, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		for(i=0; i < chunk; ++i) {
			matrix_t gx[M],gy[M];
			for(j=0; j < M; ++j) {
				gx[j] = 0;
				gy[j] = 0;
				int start_y = 1 ? -edge_half : -i;
				int end_y = i + edge_half < window - edge_half ? edge_half : (chunk - i - 1);
				for (k = start_y; k <= end_y; ++k) {
					gy[j] += buffer[(M*(i-k+half))+j]*hy[-k+edge_half];
					gx[j] += buffer[(M*(i-k+half))+j]*hx[-k+edge_half];
				}
			}

			for(j=0; j < M; ++j) {
				matrix_t sum_x = 0;
				matrix_t sum_y = 0;
				int start_x = 1 ? -edge_half :  -j;
				int end_x = M - j > edge_half ? edge_half : (M - j - 1);
				for (k = start_x; k <= end_x; ++k) {
					sum_x += gx[j+k]*hy[-k+edge_half];
					sum_y += gy[j+k]*hx[-k+edge_half];
				}
				smoothed[(M*i) + j] = sqrtf(sum_x*sum_x+sum_y*sum_y);
			}
		}

        MPI_Send(smoothed, chunk*M, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
     }

    /* Program ends */

     printf("done %d\n", world_rank);

    // Finalize the MPI environment.
     MPI_Finalize();
     return 0;
  }
