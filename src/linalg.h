#ifndef LINALG_H
#define LINALG_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TOL 1e-10
#define MAX_ITER 100

typedef struct {
  int rows;
  int cols;
  double **data;
} Matrix;

typedef struct {
  Matrix *left_matrix;
  Matrix *right_matrix;
} MatrixDecomposition;

Matrix *matrix_create(int rows, int cols);
void matrix_free(Matrix *mat);
void matrix_print(const Matrix *mat);
Matrix *matrix_copy(const Matrix *source);
Matrix *matrix_identity(int n);
Matrix *matrix_diagonal(int n, double value);
Matrix *matrix_ones(int n, int p);
Matrix *matrix_zeros(int n, int p);
Matrix *matrix_random_vector(int rows);
Matrix *matrix_multiply(const Matrix *A, const Matrix *B);
Matrix *matrix_add(const Matrix *A, const Matrix *B);
Matrix *matrix_scalar_multiply(const Matrix *mat, double scalar);
double vector_dot_product(const Matrix *v1, const Matrix *v2);
double vector_norm2(const Matrix *v);
int vector_allclose(const Matrix *v1, const Matrix *v2, double tol);
MatrixDecomposition *matrix_LU_decomposition(const Matrix *mat);
MatrixDecomposition *matrix_QR_decomposition(const Matrix *mat);
Matrix *matrix_solve_linear_system(const Matrix *A, const Matrix *b);
Matrix *matrix_inverse(const Matrix *mat);
Matrix *matrix_get_column(const Matrix *mat, int j);
void matrix_set_column(Matrix *mat, const Matrix *column_vector, int j);
Matrix *matrix_solve_lower_triangular(const Matrix *L, const Matrix *b);
Matrix *matrix_solve_upper_triangular(const Matrix *U, const Matrix *b);
Matrix *matrix_power_iteration(const Matrix *A, double tol, int max_iter);

#endif