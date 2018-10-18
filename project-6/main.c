#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "../matrix.h"

void matrix_gen_F(matrix *F, const size_t M) {
   assert(F != NULL);
   assert(F->rows == M);
   assert(F->cols == M);
   assert(F->data != NULL);
   int i, j;

   for (i = 0; i < M; i++) {
      for (j = 0; j < M; j++) {
         INDEX(F,i,j) = j % 2 ? 0 : 1089;
      }
   }
}

void matrix_gen_H(matrix *H, const size_t W) {
   assert(H != NULL);
   assert(H->rows == W);
   assert(H->cols == W);
   assert(H->data != NULL);
   int i, j;

   for (i = 0; i < W; i++) {
      for (j = 0; j < W; j++) {
         INDEX(H,i,j) = 1.0/(pow(W,2));
      }
   }
}

double percent_error(matrix *C, const size_t M) {
   double total_error, error;
   int i,j;

   total_error = 0;
   error = 0;
   #pragma omp simd collapse(2) reduction (+:total_error)
   for (i = 0; i < M; i++) {
      for (j = 0; j < M; j++) {
         error = fabs(INDEX(C,i,j) - (i+1.0)*(j+1.0)) / ((i+1.0)*(j+1.0));
         total_error += error;
      }
   }

   return (total_error * 100) / pow(M,2);
}

int main(int argc, char **argv) {
   const size_t M = 3240;
   const size_t W = 33;
   int threads;
   double start_time, end_time;

   if (argc > 1) {
      threads = atoi(argv[1]); 
      printf("Using %d threads...\n", threads);
      omp_set_num_threads(threads);
   }

   start_time = omp_get_wtime();
   printf("Starting...\n");
   matrix* F = matrix_create(M, M, NULL);
   matrix* H = matrix_create(W, W, NULL);

   printf("Generating F\n");
   matrix_gen_F(F, M);
   printf("Generating H\n");
   matrix_gen_H(H, W);

   printf("Convolving F and H\n");
   matrix* G = matrix_convolve(F, H); 
   printf("Done convolving F and H\n");

   printf("Convolving F and H\n");
   matrix_convolve(F, H); 
   printf("Done convolving F and H\n");
   
   printf("\n");
   matrix_print_some(G, 1615, 1624, 0, 10);
   printf("\n");

   int i,j;

   for (i = 0;i < F->rows; ++i) {
      for (j = 0;j < F->cols; ++j) {
         assert(INDEX(F,i,j) == INDEX(G,i,j));
      }
   }

   end_time = omp_get_wtime();

   printf("Total time: %f (sec)\n\n", (end_time-start_time));

   matrix_destroy(F);
   matrix_destroy(H);
   matrix_destroy(G);

   return 0;
}
