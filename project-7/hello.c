#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <math.h>

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

    int M,W,half,chunk;
    int i,j,k,l;
    int source,dest;
    int window,offset;
    int peek_behind,peek_ahead;
    double start,end;   

 
    // Communication buffers
    double *buffer,*result;
    // Filter and pointer used for looping
    double *filter,*p;

    M = 3240;
    W = 33;

    half = W/2;


    /* chunk is the number of rows each thread is responsible for */
    chunk = M / world_size;
    
    /* since we need information above and below the chunk we call that our
       window of rows we need to perform convolution on out chunk */
    window = chunk + (W-1);

    // Make sure that the first chunk does not undershoot the image
    peek_behind = (world_rank * chunk - half) < 0 ? 0:-half;
    // Make sure the last chunk does not overshoot the image
    peek_ahead = (world_rank * chunk + half) > M ? 0 : half;

    // We subtract half to get the stuff above us
    // offset = (world_rank * chunk - half)*M;

    filter = calloc(W*W, sizeof(double));
    assert(filter != NULL);

    result = calloc(chunk*M, sizeof(double));
    assert(result != NULL);

    // Generating filter
    for (p = &filter[(W*W)-1]; p != filter-1; p--) {
      *p = 1.0/(pow(W,2));
    }

    if (world_rank == 0) {
        double *image;

        image = calloc(M*M, sizeof(double));
        assert(image != NULL);

        // Generating image
        p = image;
        for (j = M*M; j > 0; j-=2) {
          *p = 1089;
          p += 2;
        }

        start = MPI_Wtime();

        // Start at 1 because I dont have to send to myself
        for (dest = 1; dest < world_size; dest++) {
          // We subtract half to get the stuff above us
          offset = (dest * chunk - half)*M;

          if (offset + (window*M) > M*M) {
            // Make sure that we do not send anything past our image
             window = window - half;
          }

          MPI_Send(&image[offset], window*M, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }

        // This is for the case that chunk = M
        peek_ahead = (chunk + half) > M ? 0 : half;

        for(i=0; i < chunk; ++i) {
          for(j=0; j < M; ++j) {
            for(k=-half; k <= half; ++k) {
              for(l=-half; l <= half; ++l) {
                if( (i-k >= peek_behind && i-k <= chunk+peek_ahead) && (j-l >= 0 && j-l < M) ) {
                  result[(M*i) + j] += image[M*(i-k)+(j-l)]*filter[W*(k+half)+(l+half)];
                }
              }
            }
          }
        }

        memcpy(image,result,chunk*M*sizeof(double));

        for (j = 1; j < world_size; j++) {
          source = j;
          offset = (source * chunk)*M;
          MPI_Recv(&image[offset], chunk*M, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

	end = MPI_Wtime();

	printf("\nTotal time with %d threads: %f\n\n",world_size,end-start);

        for (j = 1618; j < 1623; j++) {
          for (k = 0; k < 18; k++) {
             printf("%4.3f, ", image[M*j+k]); // Notice only 3 digits
          }
          printf("\n");
        }
	printf("\n");
      } else { // Begin not master node stuff

        buffer = calloc(window*M, sizeof(double));
        assert(buffer != NULL);

        source = 0; // Master node
        MPI_Recv(buffer, window*M, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(i=0; i < chunk; ++i) {
          for(j=0; j < M; ++j) {
            for(k=-half; k <= half; ++k) {
              for(l=-half; l <= half; ++l) {
                if( (i-k >= peek_behind && i-k <= chunk + peek_ahead) && (j-l >= 0 && j-l < M) ) {
                  result[(M*i) + j] += buffer[M*(i-k+half)+(j-l)]*filter[W*(k+half)+(l+half)];
                }
              }
            }
          }
        }

        dest = 0;
        MPI_Send(result, chunk*M, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
      }

    /* Program ends */

    // Finalize the MPI environment.
    MPI_Finalize();
}
