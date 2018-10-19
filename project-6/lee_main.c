#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define INDEX(m, row, col) \
  m->data[(m->cols) * row + (col)]

typedef double matrix_t;

struct {
	size_t rows;
	size_t cols;
	matrix_t *data;
} typedef matrix;

matrix* matrix_create(size_t rows, size_t cols, double data[cols][rows]) {
   assert(rows > 0);
   assert(cols > 0);

   matrix* m = malloc(sizeof(matrix));

   assert(m != NULL);

   m->rows = rows;
   m->cols = cols;

   m->data = (double *) calloc(rows*cols, sizeof(double));
   assert(m->data != NULL);

   // Only set data if they passed us something
   if (data != NULL) {
      memcpy(m->data, data, rows*cols*sizeof(double));
   }

   return m;
}

void matrix_destroy(matrix* m) {
   assert(m != NULL);
   assert(m->data != NULL);

   free(m->data);
   free(m);
}

void matrix_print_some(matrix* m, const size_t X1, const size_t X2, const size_t Y1, const size_t Y2) {
   int i,j;
   for (i = X1; i <= X2; i++) {
      for (j = Y1; j < Y2; j++) {
         printf("%.3f, ", INDEX(m,i,j)); // Notice only 3 digits
      }
      printf("\n");
   }
}

matrix* matrix_convolve(matrix* F, matrix* H) {

   matrix* G;
   G = matrix_create(F->rows, F->cols, NULL);

   #pragma omp parallel shared(F,H,G)
   {
     int i,j,k,m,ii,jj,kk,mm;
     int x_center, y_center, tid;

     x_center = H->cols / 2;
     y_center = H->rows / 2;
      tid = omp_get_thread_num();
      //printf("Thread %d checking in\n",tid);
      #pragma omp for
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
   }
   return G;
}

void matrix_convolve_p(matrix* F, matrix* H) {
   #pragma omp parallel shared(F,H)
   {
      int i,j,k,m,ii,jj,kk,mm,tid;
      tid = omp_get_thread_num();
      int n = omp_get_num_threads();
      int row_start, col_end;
      matrix* G;

      int x_center, y_center;

      x_center = H->cols / 2;
      y_center = H->rows / 2;

      G = matrix_create((size_t)(F->rows / n), F->cols, NULL);
      row_start = tid * (F->rows / n);

      #pragma omp exclusive
      for(i=0; i < G->rows; ++i) {
         for(j=0; j < G->cols; ++j) {
            for(k=0; k < H->rows; ++k) {
               kk = H->rows - 1 - k;
               for(m=0; m < H->cols; ++m) {
                  mm = H->cols - 1 - m;
                  ii = row_start + i + (y_center - kk);
                  jj = j + (x_center - mm);

                  // Only compute indexes inside matrix
                  if( ii >= 0 && ii < F->rows && jj >= 0 && jj < F->cols ) {
                     INDEX(G,i,j) += INDEX(F,ii,jj) * INDEX(H,kk,mm);
                  }
               }
            }
         }
      }

      #pragma omp barrier
      memcpy(&(INDEX(F,row_start,0)), &(INDEX(G,0,0)), G->rows*G->cols*sizeof(matrix_t));

      matrix_destroy(G);

 }
}

int main(int argc, char **argv) {
   const size_t M = 3240;
   const size_t W = 33;
   int threads,i,j;
   double start_time, end_time;

   if (argc > 1) {
      threads = atoi(argv[1]);
      printf("Using %d threads...\n", threads);
      omp_set_num_threads(threads);
   }

   matrix* F = matrix_create(M, M, NULL);
   matrix* H = matrix_create(W, W, NULL);

   // Generating F
   for (i = 0; i < M; i++) {
      for (j = 0; j < M; j++) {
         INDEX(F,i,j) = j % 2 ? 0 : 1089;
      }
   }

   // Generating H
   for (i = 0; i < W; i++) {
      for (j = 0; j < W; j++) {
         INDEX(H,i,j) = 1.0/(pow(W,2));
      }
   }

   // Convolving F and H using global G
   start_time = omp_get_wtime();
   matrix* G = matrix_convolve(F, H);
   end_time = omp_get_wtime();
   printf("Total time using global G: %f (sec)\n", (end_time-start_time));

   // Convolving F and H using local G
   start_time = omp_get_wtime();
   matrix_convolve_p(F, H);
   end_time = omp_get_wtime();
   printf("Total time using local G: %f (sec)\n\n", (end_time-start_time));

   printf("\nGlobal G results:\n");
   matrix_print_some(G, 1615, 1624, 0, 10);
   printf("\nLocal G results:\n");
   matrix_print_some(F, 1615, 1624, 0, 10);
   printf("\n");printf("\n");

   for (i = 0;i < F->rows; ++i) {
      for (j = 0;j < F->cols; ++j) {
         if (fabs(INDEX(F,i,j)- INDEX(G,i,j)) > .001 ) {
            printf("index %d %d %f\n",i,j,fabs(INDEX(F,i,j)- INDEX(G,i,j)));
         }
      }
   }

   matrix_destroy(F);
   matrix_destroy(H);
   matrix_destroy(G);

   return 0;
}
