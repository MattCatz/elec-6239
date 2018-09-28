#include <stdio.h>
#include <string.h>

typedef struct matrix matrix;

matrix* matrix_create(size_t rows, size_t cols, float data[cols][rows]);
void matrix_destroy(matrix* m);
void matrix_print(matrix* m);
matrix* matrix_mul(matrix* A, matrix* B);
matrix* matrix_con(matrix* F, matrix* H);

int main() {
	float A_data[4][4] = {{1,2,3,4},
	 											 {4,3,2,1},
												 {6,7,8,9},
												 {9,8,7,6}};

	float B_data[4][4] = {{4,3,2,1},
												 {1,2,3,4},
											   {4,3,2,1},
											   {1,2,3,4}};

	matrix* A = matrix_create(4, 4, A_data);
	matrix* B = matrix_create(4, 4, B_data);

	matrix* C = matrix_mul(A, B);

	printf("\nMatrix A is: \n");
	matrix_print(A);

	printf("\nMatrix B is: \n");
	matrix_print(B);

	printf("\nResult Matrix is \n");
	matrix_print(C);

	matrix_destroy(A);
	matrix_destroy(C);

  float F_data[8][8] = {{0,0,0,0,0,0,0,0},
												 {0,0,0,0,0,0,0,0},
												 {0,0,1,1,1,1,0,0},
												 {0,0,1,1,1,1,0,0},
												 {0,0,1,1,1,1,0,0},
												 {0,0,1,1,1,1,0,0},
												 {0,0,0,0,0,0,0,0},
												 {0,0,0,0,0,0,0,0}};

	float H_data[3][3] = {{1/9., 1/9., 1/9.},
												 {1/9., 1/9., 1/9.},
												 {1/9., 1/9., 1/9.}};

	matrix* F = matrix_create(8, 8, F_data);
	matrix* H = matrix_create(3, 3, H_data);

	printf("\nImage matrix is:\n");
	matrix_print(F);

	printf("\nFilter matrix is:\n");
	matrix_print(H);

	printf("\nImage after convolution is: \n");
	matrix* G = matrix_con(F, H);
	matrix_print(G);

	matrix_destroy(G);
	matrix_destroy(F);
	matrix_destroy(H);

	return 0;
}
