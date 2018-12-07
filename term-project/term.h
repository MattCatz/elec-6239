#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358
#endif

#define M 2048
#define W_SMOOTHING 15
#define W_EDGE 3
#define SIGMA 1.5

#define INDEX(m, row, col, col_length) \
m[(col_length) * (row) + (col)]

#define MPI_MATRIX_T MPI_FLOAT;
typedef float matrix_t;

void generate_guassian_2d(matrix_t *guassian) {
	int i,j;
	const int L = (W_SMOOTHING - 1)/2;
	matrix_t exponent;
	const matrix_t temp = 1 / (2*M_PI*SIGMA*SIGMA);

	for (i=0;i<W_SMOOTHING;i++) {
		for (j=0;j<W_SMOOTHING;j++) {
			exponent = ((i-L)*(i-L)+(j-L)*(j-L))/(2*SIGMA*SIGMA);
			guassian[W_SMOOTHING*i+j] = temp * exp(-exponent);
		}
	}
}

void generate_guassian(matrix_t *guassian) {
	int i;
	const int L = (W_SMOOTHING - 1)/2;
	matrix_t exponent;
	const matrix_t temp = 1 / (sqrt(2*M_PI)*SIGMA);

	for (i=0;i<W_SMOOTHING;i++) {
		exponent = ((i-L)*(i-L))/(2*SIGMA*SIGMA);
		guassian[i] = temp * exp(-exponent);
	}
}

void generate_sobel_2d(matrix_t *hx) {
	const matrix_t sobel_x[9] = {-1,0,1,-2,0,2,-1,0,1};

	memcpy(hx, sobel_x, 9*sizeof(matrix_t));
}

void generate_sobel(matrix_t *hx, matrix_t *hy) {
	const matrix_t sobel_x[3] = {1,0,-1};
	const matrix_t sobel_y[3] = {1,2,1};

	memcpy(hx, sobel_x, 3*sizeof(matrix_t));
	memcpy(hy, sobel_y, 3*sizeof(matrix_t));
}

void get_image(matrix_t *image) {
	unsigned int i;
	unsigned char in[M*M];
	matrix_t *px;
	FILE *fp;

	char file_name[] = "Leaves_noise.bin";

	fp = fopen(file_name, "r"); // read mode

	assert(fp != NULL);

	assert(M * M == fread(in, 1, M * M, fp));

	for (i=0,px=image;i<M*M;++i,++px) {
		*px = (matrix_t) in[i];
	}

	fclose(fp);
}

void save_ppm(const char *name, matrix_t *image) {
	unsigned int i;
	unsigned char out[M*M];
	matrix_t *px;
	FILE *fp;

	fp = fopen(name, "w");

	fprintf(fp, "P5\n %d %d \n 255\n", M, M);

	for (i=0,px=image;i<M*M;++i,++px) {
		out[i] = lround(*px);
	}

	fwrite(out, 1, M * M, fp);

	fclose(fp);
}

void save_g(const char *name, matrix_t *image) {
	unsigned int i;
	unsigned char out[W_SMOOTHING*W_SMOOTHING];
	matrix_t *px;
	FILE *fp;

	fp = fopen(name, "w");

	fprintf(fp, "P5\n %d %d \n 255\n", W_SMOOTHING, W_SMOOTHING);

	for (i=0,px=image;i<W_SMOOTHING*W_SMOOTHING;++i,++px) {
		out[i] = lround(*px);
	}

	fwrite(out, 1, W_SMOOTHING*W_SMOOTHING, fp);

	fclose(fp);
}
