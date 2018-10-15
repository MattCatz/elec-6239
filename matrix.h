#ifndef MATRIX_H
#define MATRIX_H

#define INDEX(m, row, col) \
  m->data[(m->cols) * row + (col)]

struct {
	size_t rows;
	size_t cols;
	double *data;
} typedef matrix;

matrix* matrix_create(size_t rows, size_t cols, double data[cols][rows]);
void matrix_destroy(matrix* m);
void matrix_print(matrix* m);
void matrix_print_some(matrix* m, const size_t X1, const size_t X2, const size_t Y1, const size_t Y2);
matrix* matrix_mul(matrix* A, matrix* B);
matrix* matrix_con(matrix* F, matrix* H);

#endif
