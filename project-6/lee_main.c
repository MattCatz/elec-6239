#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define INDEX(m, row, col) \
  m->data[((m->cols) * (row)) + (col)]

#define INDEX_M(m, row, col, cols) \
  m[((cols) * (row)) + (col)]

   const size_t M = 3240;
   const size_t W = 33;
 
typedef double matrix_t;

matrix_t* matrix_convolve(matrix_t* F, matrix_t* H) {

   matrix_t* G;
   G = malloc(M*M*sizeof(matrix_t));

   return G;
}

int main(int argc, char **argv) {
  int threads,i,j;
   double start_time, end_time;

   if (argc > 1) {
      threads = atoi(argv[1]);
      printf("Using %d threads...\n", threads);
      omp_set_num_threads(threads);
   } 

   matrix_t *F_data, *H_data, *G_data, *k;

   F_data = malloc(M*M*sizeof(matrix_t));
   H_data = malloc(W*W*sizeof(matrix_t));
   G_data = malloc(M*M*sizeof(matrix_t));

   for (k = F_data; k < &F_data[M*M]; k=k+2) {
      *k = 1089;
   }

   for (k = H_data; k < &H_data[W*W]; k++) {
      *k = 1.0/(pow(W,2));
   }

   // Convolving F and H using global G
   start_time = omp_get_wtime();

   const int half = (W/2);

   #pragma omp parallel
   {
     int i,j,k,l,i_addr,j_addr;
     int tid, offset;
     double sum;

      tid = omp_get_thread_num();
      #pragma omp for
      for(i=0; i < M; ++i) {
         for(j=0; j < M; ++j) {
	    sum = 0;
            for(k=-half; k <= half; ++k) {
               for(l=-half; l <= half; ++l) {
                  if( (i-k >= 0 && i-k < M) && (j-l >= 0 && j-l < M) ) {
                      sum += F_data[M*(i-k)+(j-l)]*H_data[W*(k+(W/2))+(l+(W/2))];
                  }
               }
            }
	    G_data[(M*i) + j] = sum;
         }
      }
   }

   for (k = G_data; k < &G_data[M*M]; k++) {
      assert(*G_data != 0);
   }

   end_time = omp_get_wtime();
   printf("Total time using global G: %f (sec)\n", (end_time-start_time));

   return 0;
}
