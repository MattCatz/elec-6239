#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#define len 4
#define M 3
#define W 3

struct {
	size_t l;
	size_t w;
	int *data;
} typedef matrix;

matrix* matrix_create(size_t rows, size_t cols) {
	assert(rows > 0);
	assert(cols > 0);

	matrix* m = malloc(sizeof(matrix));

	assert(m != NULL);

	m->l = rows;
	m->w = cols;

	m->data = (int *) calloc(rows*cols, sizeof(int));

	assert(m->data != NULL);

	return m;
}

void matrix_destroy(matrix* m) {
	assert(m != NULL);
	assert(m->data != NULL);

	free(m->data);
	free(m);
}

#define ELEM(mtx, row, col) \
  mtx->data[(col) * mtx->w + (row)]

matrix* matrix_mul(matrix* A, matrix* B) {
	assert(A != NULL);
	assert(B != NULL);
	assert(A->w == B->l);

	matrix* C = matrix_create(A->w,A->w);

	int i,j,k;
	for (i = 0; i < A->w; i++) {
		for (j = 0; j < B->l; j++) {
			for (k = 0; k < A->w; k++) {
				ELEM(C,i,j) += ELEM(A,i,k)*ELEM(B,k,j);
			}
		}
	}

	return C;
}

void mul(int A[][len], int B[][len], int C[][len]) {
	int i,j,k;
	for (i = 0; i < len; i++) {
		for (j = 0; j < len; j++) {
			C[i][j] = 0;
			for (k = 0; k < len; k++) {
				C[i][j] += A[i][k]*B[k][j];
			}
		}
	}
}

void convilution(int F[][M], int H[][W], int G[][M]) {
	int i,j,m,n,mm,nn,ii,jj;
	int kCenterX = W / 2;
  int kCenterY = W / 2;

	for(i=0; i < M; ++i) {
		for(j=0; j < M; ++j) {
			for(m=0; m < W; ++m) {
					mm = W - 1 - m;      // row index of flipped kernel

					for(n=0; n < W; ++n) {
					nn = W - 1 - n;  // column index of flipped kernel

					// index of input signal, used for checking boundary
					ii = i + (kCenterY - mm);
					jj = j + (kCenterX - nn);

					// ignore input samples which are out of bound
					if( ii >= 0 && ii < M && jj >= 0 && jj < M )
						G[i][j] += F[ii][jj] * H[mm][nn];
				}
			}
		}
	}
}

int main() {
	int i, j;

	int A[][len] = {{1,2,3,4}, {4,3,2,1}, {6,7,8,9}, {9,8,7,6}};
	int C[len][len] = {0};

	mul(A, A, C);

	matrix* ASTRUCT = matrix_create(4, 4);
	memcpy(ASTRUCT->data, A, 16*sizeof(int));

	matrix* CSTRUCT = matrix_mul(ASTRUCT, ASTRUCT);

	printf("Result is \n");
	for (i = 0; i < CSTRUCT->l; i++) {
		for (j = 0; j < CSTRUCT->w; j++) {
			printf("%d ", ELEM(CSTRUCT,i,j));//C[i][j]);
		}
		printf("\n");
	}


	int F[][M] = {{1,2,3},{4,5,6},{7,8,9}};
	int H[][W] = {{-1, -2, -1},{0,0,0},{1,2,1}};
	int G[M][M] = {0};

	convilution(F, H, G);

	printf("Result is \n");
	for (i = 0; i < M; i++) {
		for (j = 0; j < M; j++) {
			printf("%d ", G[i][j]);
		}
		printf("\n");
	}

	return 0;
}
