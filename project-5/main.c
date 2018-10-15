#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "../matrix.h"

void matrix_gen_a(matrix *A, const size_t M) {
  assert(A != NULL);
  assert(A->rows == M);
  assert(A->cols == M);
  assert(A->data != NULL);
  int i, j;

  #pragma omp parallel shared(A) private(i,j) 
  {
     #pragma omp for schedule(static)
     for (i = 0; i < M; i++) {
       for (j = 0; j < M; j++) {
         INDEX(A,i,j) = ((i+1.0)*(j+1.0)) / M;
       }
     }
   }
}

void matrix_gen_b(matrix *B, const size_t M) {
  assert(B != NULL);
  assert(B->rows == M);
  assert(B->cols == M);
  assert(B->data != NULL);
  int i, j;

  #pragma omp parallel shared(B) private(i,j) 
  {
    #pragma omp for schedule(static)
    for (i = 0; i < M; i++) {
      for (j = 0; j < M; j++) {
        INDEX(B,i,j) = (j+1.0)/(i+1.0);
      }
    }
  }
}

double percent_error(matrix *C, const size_t M) {
  double total_error, error;
  int i,j;

  total_error = 0;
  error = 0;
  #pragma omp parallel shared(C,total_error) private(error, i, j)
  {
    #pragma omp for reduction (+:total_error)
    for (i = 0; i < M; i++) {
      for (j = 0; j < M; j++) {
        error = fabs(INDEX(C,i,j) - (i+1.0)*(j+1.0)) / ((i+1.0)*(j+1.0));
        total_error += error;
      }
    }
  }
  
  return (total_error * 100) / pow(M,2);
}

int main() {
  const size_t M = 3240;
  double start_time, end_time;

  #ifdef THREADS
  printf("Using %d threads...\n", THREADS);
  omp_set_num_threads(THREADS);
  #endif
 
  start_time = omp_get_wtime();
  printf("Starting...\n");
  matrix* A = matrix_create(M, M, NULL);
  matrix* B = matrix_create(M, M, NULL);

  printf("Generating A\n");
  matrix_gen_a(A, M);
  printf("Generating B\n");
  matrix_gen_b(B, M);

  printf("Multiplying  A and B\n");
  matrix* C = matrix_mul(A, B); 
  printf("Done multiplying A and B\n");

  printf("Percent error is %.6f\n", percent_error(C, M));

  printf("\n");
  matrix_print_some(C, 1617, 1623, 1617, 1623);
  printf("\n");

  matrix_destroy(A);
  matrix_destroy(B);
  matrix_destroy(C);

  end_time = omp_get_wtime();

  printf("Total time: %f (sec)\n", (end_time-start_time));

  return 0;
}
