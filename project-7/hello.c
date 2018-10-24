#include <stdlib.h>
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

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);

    /* Program begins */

    int M,W,source,dest;
    int j,k,chunk,window,offset;
    M = 40;
    W = 3;

    // Communication buffers
    double *buffer,*result;

    MPI_Request request ;
    MPI_Status status ;

    // Determine chunk and padding
    chunk = M / world_size;
    window = chunk + (W-1);

    double *filter,*i;

    filter = calloc(W*W, sizeof(double));
    assert(filter != NULL);

    // Generating filter
    for (i = &filter[(W*W)-1]; i != filter-1; i--) {
      *i = 1.0/(pow(W,2));
    }

    if (world_rank == 0) {
        double *image;
        double *i;

        printf("Thread 0 setting up some stuff\n");

        image = calloc(M*M, sizeof(double));
        assert(image != NULL);

        // Generating image
        for (i = &image[(M*M)-1]; i != image-1; i--) {
          *i = 1089;//j % 2 ? 0 : 1089;
        }

        // Generating filter
        // for (i = filter + (W*W); i != filter; i--) {
        //   *i = 1.0/(pow(W,2));
        // }

        // Start at 1 because I dont have to send to myself
        for (dest = 1; dest < world_size; dest++) {
          offset = (dest * chunk);
          printf("Sending out partition %d to thread %d\n",offset,dest);
          MPI_Send(&image[offset], 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD); //image
        }

        //convolve(image, filter);
        printf("Master thread doing convolution\n");

        for (j = 1; j < world_size; j++) {
          source = j,
          MPI_Recv(&k, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          printf("Master thread %d got back %d\n", world_rank, k);
        }

        for (j = 0; j < 40; j++) {
          for (k = 0; k < 40; k++) {
             printf("%.0f, ", INDEX(image,M,j,k)); // Notice only 3 digits
          }
          printf("\n");
        }
      } else {
        buffer = calloc(window*M, sizeof(double));
        assert(buffer != NULL);

        result = calloc(chunk*M, sizeof(double));
        assert(result != NULL);

        int b;

        printf("Thread %d waiting on partition\n", world_rank);
        source = 0; // Master node
        MPI_Recv(buffer, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Thread %f got partition\n", *buffer);
        // for (j = 0; j < 10; j++) {
        //   for (k = 0; k < 10; k++) {
        //      //printf("%.3f, ", INDEX(buffer,M,j,k)); // Notice only 3 digits
        //   }
        //   printf("\n");
        // }

        dest = 0;
        //MPI_Send(&b, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
        MPI_Send(&b, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        printf("Thread %d done\n", world_rank);
      }

    /* Program ends */

    // Finalize the MPI environment.
    MPI_Finalize();
}
