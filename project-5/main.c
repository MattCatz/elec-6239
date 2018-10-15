#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <omp.h>
#include "../matrix.h"

void matrix_gen_a(matrix *A, const size_t M) {
  assert(A != NULL);
  assert(A->rows == M);
  assert(A->cols == M);
  assert(A->data != NULL);
  int i, j;
  for (i = 0; i < M; i++) {
    for (j = 0; j < M; j++) {
      INDEX(A,i,j) = (i+1)*(j+1)*(1/M);
    }
  }
}

void matrix_gen_b(matrix *B, const size_t M) {
  assert(B != NULL);
  assert(B->rows == M);
  assert(B->cols == M);
  assert(B->data != NULL);
  int i, j;
  for (i = 0; i < M; i++) {
    for (j = 0; j < M; j++) {
      INDEX(B,i,j) = (i+1)/(j+1);
    }
  }
}

void matrix_gen_sanity(matrix *C, const size_t M) {
  assert(C != NULL);
  assert(C->rows == M);
  assert(C->cols == M);
  assert(C->data != NULL);
  int i,j;
  for (i = 0; i < M; i++) {
    for (j = 0; j < M; j++) {
      INDEX(C,i,j) = (i+1)*(j+1);
    }
  }
}

int main() {
	const size_t M = 3240;
//  const size_t M = 3240;
  omp_set_num_threads(10); 
 
  printf("Starting...\n");
	matrix* A = matrix_create(M, M, NULL);
	matrix* B = matrix_create(M, M, NULL);
	matrix* sanity = matrix_create(M, M, NULL);

  printf("Generating A\n");
  matrix_gen_a(A, M);
  printf("Generating B\n");
  matrix_gen_b(B, M);
  printf("Generating sanity\n");
  matrix_gen_sanity(sanity, M);

  printf("Multiplying  A and B\n");
	matrix* C = matrix_mul(A, B); 
  printf("Done multiplying A and B\n");

	matrix_destroy(A);
	matrix_destroy(B);
	matrix_destroy(C);
	matrix_destroy(sanity);

	return 0;
}
