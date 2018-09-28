#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

#define INDEX(m, row, col) \
  m->data[(m->cols) * row + (col)]

struct {
	size_t rows;
	size_t cols;
	float *data;
} typedef matrix;

matrix* matrix_create(size_t rows, size_t cols, float data[cols][rows]) {
	assert(rows > 0);
	assert(cols > 0);

	matrix* m = malloc(sizeof(matrix));

	assert(m != NULL);

	m->rows = rows;
	m->cols = cols;

	m->data = (float *) calloc(rows*cols, sizeof(float));
	assert(m->data != NULL);

	// Only set data if they passed us something
	if (data != NULL) {
		memcpy(m->data, data, rows*cols*sizeof(float));
	}

	return m;
}

void matrix_destroy(matrix* m) {
	assert(m != NULL);
	assert(m->data != NULL);

	free(m->data);
	free(m);
}

void matrix_print(matrix* m) {
	int i,j;
	assert(m != NULL);
	for (i = 0; i < m->rows; i++) {
		for (j = 0; j < m->cols; j++) {
			printf("%.3f, ", INDEX(m,i,j)); // Notice only 3 digits
		}
		printf("\n");
	}
}

matrix* matrix_mul(matrix* A, matrix* B) {
	assert(A != NULL);
	assert(B != NULL);
	assert(A->cols == B->rows);

	matrix* C = matrix_create(A->rows, B->cols, NULL);

	int row,col,k;
	for (row = 0; row < A->rows; row++) {
		for (col = 0; col < B->cols; col++) {
			for (k = 0; k < A->cols; k++) {
				INDEX(C,row,col) += INDEX(A,row,k)*INDEX(B,k,col);
			}
		}
	}

	return C;
}

matrix* matrix_con(matrix* F, matrix* H) {
	int i,j,k,m,ii,jj,kk,mm;
	int x_center, y_center;
	matrix* G;

	assert(F != NULL);
	assert(H != NULL);

	G = matrix_create(F->rows, F->cols, NULL);

	x_center = H->cols / 2;
  y_center = H->rows / 2;

	for(i=0; i < F->rows; ++i) {
		for(j=0; j < F->cols; ++j) {
			for(k=0; k < H->rows; ++k) {
					kk = H->rows - 1 - k;
					for(m=0; m < H->cols; ++m) {
						mm = H->cols - 1 - m;
						ii = i + (y_center - kk);
						jj = j + (x_center - mm);

						// Only compute indexes inside matrix
						if( ii >= 0 && ii < F->rows && jj >= 0 && jj < F->cols ) {
							INDEX(G,i,j) += INDEX(F,ii,jj) * INDEX(H,kk,mm);
						}
				}
			}
		}
	}
	return G;
}
