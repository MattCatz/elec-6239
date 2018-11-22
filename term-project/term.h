#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#define M 2048
#define W_SMOOTHING 15
#define W_EDGE 3
#define SIGMA 1.5f

#define INDEX(m, row, col, col_length) \
m[(col_length) * (row) + (col)]

#define MPI_MATRIX_T MPI_FLOAT;
typedef float matrix_t;

void generate_guassian(matrix_t *f) {
	unsigned int i,j;
	matrix_t exponent;
	const matrix_t L = (W_SMOOTHING - 1)*.5;
	const matrix_t temp = 1 / (2*M_PI*SIGMA*SIGMA);

	for (i=0;i<W_SMOOTHING;i++) {
		for (j=0;j<W_SMOOTHING;j++) {
			exponent = ((i-L)*(i-L) + (j-L)*(j-L))/(2*SIGMA*SIGMA);
			INDEX(f,i,j,W_SMOOTHING) = temp * exp(-exponent);
		}
	}
}

void generate_sobel(matrix_t *hx, matrix_t *hy) {
	const matrix_t sobel_x[9] = {-1,0,1,-2,0,2,-1,0,1};
	const matrix_t sobel_y[9] = {-1,-2,-1,0,0,0,1,2,1};

	memcpy(hx, sobel_x, 9*sizeof(matrix_t));
	memcpy(hy, sobel_y, 9*sizeof(matrix_t));
}

void get_image(matrix_t *image) {
	unsigned int i,j;
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

void save_ppm(char *name, matrix_t *image) {
	unsigned int i,j;
	unsigned char out[M*M];
	matrix_t *px;
	FILE *fp;

	fp = fopen(name, "w");

	fprintf(fp, "P5\n %d %d \n 255 \n", M, M);

	for (i=0,px=image;i<M*M;++i,++px) {
		out[i] = lround(*px);
	}

	fwrite(out, 1, M * M, fp);

	fclose(fp);
}

// int main() {
// 	char ch;
// 	FILE *fp;
// 	unsigned char *final_image;
// 	matrix_t *image, *smoothing, *hx, *hy;
// 	unsigned int i, j;

// 	image = calloc(M * M, sizeof(matrix_t));
// 	smoothing = calloc(W_SMOOTHING*W_SMOOTHING, sizeof(matrix_t));
// 	hx = calloc(W_EDGE*W_EDGE, sizeof(matrix_t));
// 	hy = calloc(W_EDGE*W_EDGE, sizeof(matrix_t));
// 	final_image = calloc(M * M, sizeof(char));

// 	get_image(image);

// 	// char file_name[] = "Leaves_noise.bin";

// 	// //
// 	// // Read file from disk
// 	// //

// 	// fp = fopen(file_name, "r"); // read mode

// 	// if (fp == NULL) {
// 	// 	perror("Error while opening the file.\n");
// 	// 	exit(EXIT_FAILURE);
// 	// }

// 	// assert(M * M == fread(final_image, 1, M * M, fp));

// 	// fclose(fp);

// 	//
// 	// Done reading file from disk
// 	//

// 	generate_guassian(smoothing);
// 	generate_sobel(hx,hy);

// 	//
// 	// Lets try to write this back to an image to see what it looks like
// 	//

// 	// fp = fopen("Leaves_noise.ppm", "w");

// 	// fprintf(fp, "P5\n %d %d \n 255 \n", M, M);

// 	// fwrite(final_image, 1, M * M, fp);

// 	// fclose(fp);

// 	save_ppm("Leaves_noise.ppm", image);

// 	return 0;
// }