#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <math.h>

#define INDEX(m, cols, row, col) \
  m[(cols) * row + (col)]

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    /* Program begins */

    int M,W,source,dest;
    int j,k,chunk,window,offset;
    int half;
    int i,l;
    M = 3240;
    W = 33;

    half = W/2;

    // Communication buffers
    double *buffer,*result;

    MPI_Request request ;
    MPI_Status status ;

    /* chunk is the number of rows each thread is responsible for */
    chunk = M / world_size;
    /* since we need information above and below the chunk we call that our
       window of rows we need to perform convolution on out chunk */
    window = chunk + (W-1);

    // We subtract half to get the stuff above us
    offset = (world_rank * chunk - half)*M;

    double *filter,*p;

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

        printf("Thread 0 setting up some stuff\n");

        image = calloc(M*M, sizeof(double));
        assert(image != NULL);

        // Generating image
        p = image;
        for (j = M*M; j > 0; j-=2) {
          *p = 1089;
          p += 2;
        }

        printf("Thread 0 setting up some stuff %d\n", M*M);


        //double test [64] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

         // for (j = 0; j < 64; j++) {
         //   image[j] = test[j];
         // }

        // Start at 1 because I dont have to send to myself
        for (dest = 1; dest < world_size; dest++) {
          // We subtract half to get the stuff above us
          offset = (dest * chunk - half)*M;

          if (offset + (window*M) > M*M) {
            // We have over shot our image buffer
            // TODO tell the thread it has a smaller window
            printf("Window over shoots the image\n");
            window = window - half;
          }

          printf("Sending out partition %d to thread %d %d\n",offset,dest,window*M);
          MPI_Send(&image[offset], window*M, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD); //image
        }

        // We do not get this third print statment so probably above this ^^
        printf("Thread 0 setting up some stuff\n");


        for(i=0; i < chunk; ++i) {
          for(j=0; j < M; ++j) {
            for(k=-half; k <= half; ++k) {
              for(l=-half; l <= half; ++l) {
                if( (i-k >= 0 && i-k <= chunk+half) && (j-l >= 0 && j-l < M) ) {
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

        for (j = 1615; j < 1624; j++) {
          for (k = 0; k < 10; k++) {
             printf("%2.3f, ", INDEX(image,M,j,k)); // Notice only 3 digits
          }
          printf("\n");
        }
      } else { // Begin not master node stuff

        buffer = calloc(window*M, sizeof(double));
        assert(buffer != NULL);

        source = 0; // Master node
        MPI_Recv(buffer, window*M, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        /* Note that right now the bottom partition looks past the bottom of the
           chunk but because it doesnt look past window and we zero it all out
           there is no effect */

        for(i=0; i < chunk; ++i) {
          for(j=0; j < M; ++j) {
            for(k=-half; k <= half; ++k) {
              for(l=-half; l <= half; ++l) {
                if( (i-k >= -half && i-k <= chunk + half) && (j-l >= 0 && j-l < M) ) {
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
