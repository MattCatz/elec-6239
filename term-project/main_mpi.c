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

    int half,chunk;
    int i,j,k,l;
    int source,dest;
    int window,offset;
    int peek_behind,peek_ahead;
    double start,end;   

 
    // Communication buffers
    matrix_t *buffer,*result;
    // Filter and pointer used for looping
    matrix_t *filter,*p;

    half = W_SMOOTHING/2;


    /* chunk is the number of rows each thread is responsible for */
    chunk = M / world_size;
    
    /* since we need information above and below the chunk we call that our
       window of rows we need to perform convolution on out chunk */
    window = chunk + (W_SMOOTHING-1);

    // Make sure that the first chunk does not undershoot the image
    peek_behind = (world_rank * chunk - half) < 0 ? 0:-half;
    // Make sure the last chunk does not overshoot the image
    peek_ahead = (world_rank * chunk + half) > M ? 0 : half;

    // We subtract half to get the stuff above us
    // offset = (world_rank * chunk - half)*M;

    filter = calloc(W_SMOOTHING*W_SMOOTHING, sizeof(matrix_t));
    assert(filter != NULL);

    result = calloc(chunk*M, sizeof(matrix_t));
    assert(result != NULL);


    // Generating filter
    generate_guassian(filter);

    if (world_rank == 0) {
        matrix_t *image;

        image = calloc(M*M, sizeof(matrix_t));
        assert(image != NULL);

        // Generating image
        get_image(image);

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

        // This is for the case that chunk = M
        peek_ahead = (chunk + half) > M ? 0 : half;

        for(i=0; i < chunk; ++i) {
          for(j=0; j < M; ++j) {
            for(k=-half; k <= half; ++k) {
              for(l=-half; l <= half; ++l) {
                if( (i-k >= peek_behind && i-k < chunk+peek_ahead) && (j-l >= 0 && j-l < M) ) {
                	if (M*(i-k)+(j-l) >= M*M) printf("%d i %d j %d k %d l %d chunk %d\n", M*(i-k)+(j-l),i,j,k,l,chunk);
                  result[(M*i) + j] += image[M*(i-k)+(j-l)]*filter[W_SMOOTHING*(k+half)+(l+half)];
                }
              }
            }
          }
        }

        memcpy(image,result,chunk*M*sizeof(matrix_t));

        for (j = 1; j < world_size; j++) {
          source = j;
          offset = (source * chunk)*M;
          MPI_Recv(&image[offset], chunk*M, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

	end = MPI_Wtime();

	printf("\nTotal time with %d threads: %f\n\n",world_size,end-start);
	save_ppm("Leaves_denoise.ppm", image);
      } else { // Begin not master node stuff

        buffer = calloc(window*M, sizeof(matrix_t));
        assert(buffer != NULL);

        source = 0; // Master node
        MPI_Recv(buffer, window*M, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(i=0; i < chunk; ++i) {
          for(j=0; j < M; ++j) {
            for(k=-half; k <= half; ++k) {
              for(l=-half; l <= half; ++l) {
                if( (i-k >= peek_behind && i-k <= chunk + peek_ahead) && (j-l >= 0 && j-l < M) ) {
                  result[(M*i) + j] += buffer[M*(i-k+half)+(j-l)]*filter[W_SMOOTHING*(k+half)+(l+half)];
                }
              }
            }
          }
        }

        dest = 0;
        MPI_Send(result, chunk*M, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      }

    /* Program ends */

    // Finalize the MPI environment.
    MPI_Finalize();
}
