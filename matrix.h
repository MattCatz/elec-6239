#ifndef MATRIX_H
#define MATRIX_H

#define INDEX(m, row, col) \
  m->data[(m->cols) * row + (col)]

struct {
	size_t rows;
	size_t cols;
	float *data;
} typedef matrix;

matrix* matrix_create(size_t rows, size_t cols, float data[cols][rows]);
void matrix_destroy(matrix* m);
void matrix_print(matrix* m);
matrix* matrix_mul(matrix* A, matrix* B);
matrix* matrix_con(matrix* F, matrix* H);

#endif
