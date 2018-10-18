#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "matrix.h"
#include <omp.h>


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

void matrix_print_some(matrix* m, const size_t X1, const size_t X2, const size_t Y1, const size_t Y2) {
   assert(X1 < X2);
   assert(Y1 < Y2);
   assert(X2 <= m->rows);
   assert(Y2 <= m->cols);
   int i,j;
   assert(m != NULL);
   for (i = X1; i < X2; i++) {
      for (j = Y1; j < Y2; j++) {
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

   int row,col,k,tid;
#pragma omp parallel shared(A,B,C) private(row,col,k,tid) 
   {
      tid = omp_get_thread_num();
      printf("Thread %d checking in\n",tid);
#pragma omp for collapse(2)
      for (row = 0; row < A->rows; row++) {
         //printf("Thread=%d did row=%d\n",tid,row);
         for (col = 0; col < B->cols; col++) {
            for (k = 0; k < A->cols; k++) {
               INDEX(C,row,col) += INDEX(A,row,k)*INDEX(B,k,col);
            }
         }
      }
   }

   return C;
}

matrix* matrix_convolve(matrix* F, matrix* H) {
   assert(F != NULL);
   assert(H != NULL);

   int i,j,k,m,ii,jj,kk,mm;
   int x_center, y_center, tid;

   x_center = H->cols / 2;
   y_center = H->rows / 2;

   matrix* G;
   G = matrix_create(F->rows, F->cols, NULL);

   #pragma omp parallel shared(F,H,G,x_center,y_center) private(i,j,k,m,ii,jj,kk,mm) 
   {  
      tid = omp_get_thread_num();
      printf("Thread %d checking in\n",tid);
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
   assert(F != NULL);
   assert(H != NULL);

   int i,j,k,m,ii,jj,kk,mm;
   int x_center, y_center, tid;

   x_center = H->cols / 2;
   y_center = H->rows / 2;

   matrix* G;
   G = matrix_create(F->rows, F->cols, NULL);

   #pragma omp parallel shared(F,H,G,x_center,y_center) private(i,j,k,m,ii,jj,kk,mm) 
   {  
      tid = omp_get_thread_num();
      int n = omp_get_num_threads();
      printf("Thread %d checking in\n",tid);
      int col_start, col_end;
      matrix* G;

      // Do this to pickup the undivisible part of the matrix
      matrix_create(F->rows, (F->cols / n), NULL);
      col_start = tid * n;
      col_end = col_start + n - 1;
      #pragma omp for 
      for(i=0; i < G->rows; i++) {
         for(j=col_start; j <= col_end; j++) {
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

     #pragma omp for
     for(i = 0; i < G->rows; i++) {
       for(j = col_start; j <= col_end; j++) {
          INDEX(F,i,j) = INDEX(G,i,j);
     }
   }
 }
}
