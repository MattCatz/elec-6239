#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <math.h>

#include "term.h"

int main(int argc, char** argv) {
    // Initialize the MPI environment
	MPI_Init(NULL, NULL);

    // Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    /* Program begins */

	int half,edge_half,chunk;
	int i,j,k,l;
	int source,dest;
	int window,offset;
	int peek_behind,peek_ahead;
	double start,end;
	matrix_t gx,gy;  


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

    // Make sure that the first chunk does not undershoot the image
	peek_behind = (world_rank * chunk - half) < 0 ? 0: -half;
    // Make sure the last chunk does not overshoot the image
	peek_ahead = (world_rank * chunk + half) > M ? 0 : half;

    // We subtract half to get the stuff above us
    // offset = (world_rank * chunk - half)*M;

	filter = calloc(W_SMOOTHING, sizeof(matrix_t));
	assert(filter != NULL);

	hx = calloc(W_EDGE*W_EDGE, sizeof(matrix_t));
	assert(hx != NULL);

	hy = calloc(W_EDGE*W_EDGE, sizeof(matrix_t));
	assert(hy != NULL);

	edge = calloc(chunk*M, sizeof(matrix_t));
	assert(edge != NULL);

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

		/* Do the row wise de-noise convolution */

		for(i=0; i < chunk; ++i) {
			for(j=0; j < M; ++j) {
				matrix_t sum = 0;
				int start_y = i - half >= 0 ? -half : -i;
				int end_y = M - i > half ? half : (M - i - 1);
				for (k = start_y; k <= end_y; ++k) {
					sum += image[(M*(i+k))+j]*filter[-k+half];
				}
				smoothed[(M*i) + j] = sum;
			}
		}

		/* Do the column wise de-noise convolution */

		for(i=0; i < chunk; ++i) {
			for(j=0; j < M; ++j) {
				matrix_t sum = 0;
				int start_x = j - half >= 0 ? -half :  -j;
				int end_x = M - j > half ? half : (M - j - 1);
				for (k = start_x; k <= end_x; ++k) {
					sum += smoothed[(M*i)+j+k]*filter[-k+half];
				}
				image[(M*i) + j] = sum;
			}
		}

		// for(i=0; i < chunk; ++i) {
		// 	for(j=0; j < M; ++j) {
		// 		matrix_t sum = 0;
		// 		int start_y = i - half >= 0 ? -half : -i;
		// 		int end_y = M - i > half ? half : (M - i - 1);
		// 		for (k = start_y; k <= end_y; ++k) {
		// 			sum += image[(M*(i+k))+j]*hy[-k+half];
		// 		}
		// 		smoothed[(M*i) + j] = sum;
		// 	}
		// }

		// for(i=0; i < chunk; ++i) {
		// 	for(j=0; j < M; ++j) {
		// 		matrix_t sum = 0;
		// 		int start_x = j - edge_half >= 0 ? -edge_half :  -j;
		// 		int end_x = M - j > edge_half ? edge_half : (M - j - 1);
		// 		for (k = start_x; k <= end_x; ++k) {
		// 			sum += smoothed[(M*i)+j+k]*hx[-k+half];
		// 		}
		// 		image[(M*i) + j] = sum;
		// 	}
		// }


//       // Make sure that the first chunk does not undershoot the image
// 		peek_behind = (world_rank * chunk - edge_half) < 0 ? 0:-edge_half;
//     	        // This is for the case that chunk = M
// 		peek_ahead = (chunk + edge_half) > M ? 0 : edge_half;

// 		for(i=0; i < chunk; ++i) {
// 			for(j=0; j < M; ++j) {
// 				gx = 0;
// 				gy = 0;
// 				for(k=-edge_half; k <= edge_half; ++k) {
// 					for(l=-edge_half; l <= edge_half; ++l) {
// 						// TODO Figure out if this is the right limits
// 						if( (i-k >= peek_behind && i-k < chunk+peek_ahead) && (j-l >= 0 && j-l < M) ) {
// 							gx += smoothed[M*(i-k)+(j-l)]*hx[W_EDGE*(k+edge_half)+(l+edge_half)];
// 							gy += smoothed[M*(i-k)+(j-l)]*hy[W_EDGE*(k+edge_half)+(l+edge_half)];
// 						}
// 					}
// 				}
// 				image[(M*i) + j] = sqrtf(gx*gx+gy*gy);
// 			}
// }

		printf("done\n");

		for (j = 1; j < world_size; j++) {
			source = j;
			offset = (source * chunk)*M;
			MPI_Recv(&image[offset], chunk*M, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		printf("done\n");


		end = MPI_Wtime();

		printf("\nTotal time with %d threads: %f\n\n",world_size,end-start);
		save_ppm("Leaves_blur.ppm", image);

		printf("done\n");

      } else { // Begin not master node stuff

      	buffer = calloc(window*M, sizeof(matrix_t));
      	assert(buffer != NULL);

        source = 0; // Master node
        MPI_Recv(buffer, window*M, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for(i=0; i < chunk; ++i) {
			// printf("%i\n", i);
			for(j=0; j < M; ++j) {
				matrix_t sum = 0;
				int start_y = 1 ? -half : -i;
				int end_y = i + half < window - half ? half : (chunk - i - 1);
				for (k = start_y; k <= end_y; ++k) {
					if (i >= chunk - 1 && j < 10) printf("%d %d %d\n", k, start_y,end_y);
					sum += buffer[(M*(i-k+half))+j]*filter[half-k];
				}
				smoothed[(M*i) + j] = sum;
			}
		}

		for(i=0; i < chunk; ++i) {
			for(j=0; j < M; ++j) {
				matrix_t sum = 0;
				int start_x = j - half >= 0 ? -half :  -j;
				int end_x = M - j > half ? half : (M - j - 1);
				for (k = start_x; k <= end_x; ++k) {
					sum += smoothed[(M*i)+j+k]*filter[half-k];
				}
				buffer[(M*i) + j] = sum;
			}
		}

        dest = 0;
        MPI_Send(buffer, chunk*M, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
     }

    /* Program ends */

     printf("done %d\n", world_rank);

    // Finalize the MPI environment.
     MPI_Finalize();
     return 0;
  }
