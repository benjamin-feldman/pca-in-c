#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

// macro that will print a header containing the name of the matrix
// uses the # operator of the C preprocessor
#define PRINT_MATRIX(mat) printf("Matrix %s:\n", #mat); print_matrix(mat)
#define TOL 1e-10 // tolerance for num algos
#define MAX_ITER 100

// basic 2-D Matrix struct
typedef struct {
  int rows;
  int cols;
  double **data;
} Matrix;

// ad-hoc struct used to hold a LU decomposition
typedef struct {
  Matrix *L;
  Matrix *U;
} LUDecomposition;

void print_matrix(const Matrix *mat);
Matrix *matmul(const Matrix *A, const Matrix *B);
Matrix *malloc_mat(int rows, int cols);
void free_matrix(Matrix *mat);
Matrix *identity(int n);
Matrix *diag(int n, double value);
LUDecomposition *LUdecompose(const Matrix *mat);

void print_matrix(const Matrix *mat) {
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      printf("%8.2f ", mat->data[i][j]);
    }
    printf("\n");
  }
}

Matrix *malloc_mat(int rows, int cols) {
  Matrix *mat = malloc(sizeof(Matrix));
  mat->rows = rows;
  mat->cols = cols;
  mat->data = malloc(rows * sizeof(double *));
  for (int i = 0; i < rows; i++) {
    mat->data[i] = malloc(cols * sizeof(double));
  }
  return mat;
}

void free_matrix(Matrix *mat) {
  for (int i = 0; i < mat->rows; i++) {
    free(mat->data[i]);
  }
  free(mat->data);
}

Matrix *matmul(const Matrix *A, const Matrix *B) {
  // returns a pointer to a matrix, so that it can also return NULL
  // in C, NULL is defined as ((void*)0)
  if (A->cols != B->rows) {
    printf("Error: Matrices do not have matching dimensions.\n");
    return NULL;
  }

  Matrix *C = malloc_mat(A->rows, B->cols);

  for (int i = 0; i < A->rows; i++) {
    for (int j = 0; j < B->cols; j++) {
      double Cij = 0;
      for (int k = 0; k < A->cols; k++) {
        Cij += A->data[i][k] * B->data[k][j];
      }
      C->data[i][j] = Cij;
    }
  }
  return C;
}

Matrix *matsum(const Matrix *A, const Matrix *B) {
  if (A->rows != B->rows || A->cols != B->cols) {
    printf("Error: Matrices do not have matching dimensions.\n");
    return NULL;
  }

  int n = A->rows;
  int p = A->cols;

  Matrix *C = malloc_mat(n, p);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < p; j++)
      C->data[i][j] = A->data[i][j] + B->data[i][j];
  return C;
}

Matrix *identity(int n) {
  Matrix *m = malloc_mat(n, n);
  for (int i = 0; i < n; i++)
    m->data[i][i] = 1.;
  return m;
}

Matrix *diag(int n, double value) {
  Matrix *m = malloc_mat(n, n);
  for (int i = 0; i < n; i++)
    m->data[i][i] = value;
  return m;
}

Matrix *ones(int n, int p) {
  Matrix *m = malloc_mat(n, p);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < p; j++)
      m->data[i][j] = 1;
  return m;
}

Matrix *zeros(int n, int p) {
  Matrix *m = malloc_mat(n, p);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < p; j++)
      m->data[i][j] = 0;
  return m;
}

Matrix *solve_lower_triangular(const Matrix *L, const Matrix *b) {
  // solve Lx = b, for L a lower triangular n*n matrix
  // using forward substitution
  int n = b->rows;
  if (n != L->cols) {
    printf("Error: Matrix must be square.\n");
    return NULL;
  }
  Matrix *x = malloc_mat(n, 1);

  for (int i = 0; i < n; i++) {
    double xi = b->data[i][0];
    for (int j = 0; j < i - 1; j++) {
      xi -= L->data[i][j];
    }
    if (L->data[i][i] == 0) {
      printf("Error: The system is not solvable since the matrix is singular.");
      free_matrix(x);
      return NULL;
    }
    xi /= L->data[i][i];
    x->data[i][0] = xi;
  }

  return x;
}

Matrix *solve_upper_triangular(const Matrix *U, const Matrix *b) {
  // solve Lx = b, for L a upper triangular n*n matrix
  // using backward substitution
  int n = b->rows;
  if (n != U->cols) {
    printf("Error: Matrix must be square.\n");
    return NULL;
  }
  Matrix *x = malloc_mat(n, 1);

  for (int i = n - 1; i >= 0; i--) {
    double xi = b->data[i][0];
    for (int j = i + 1; j < n; j++) {
      xi -= U->data[i][j];
    }
    if (U->data[i][i] == 0) {
      printf("Error: The system is not solvable since the matrix is singular.");
      free_matrix(x);
      return NULL;
    }
    xi /= U->data[i][i];
    x->data[i][0] = xi;
  }
  return x;
}

LUDecomposition *LUdecompose(const Matrix *mat) {
  // computes LU decomposition using the Doolittle algorithm
  int n = mat->rows;
  if (n != mat->cols) {
    printf("Error: Matrix must be square for LU decomposition.\n");
    return NULL;
  }

  Matrix *L = malloc_mat(n, n);
  Matrix *U = malloc_mat(n, n);

  for (int i = 0; i < n; i++) {
    // upper triangular
    for (int k = i; k < n; k++) {
      double sum = 0;
      for (int j = 0; j < i; j++) {
        sum += L->data[i][j] * U->data[j][k];
      }
      U->data[i][k] = mat->data[i][k] - sum;
    }

    // lower triangular
    for (int k = i; k < n; k++) {
      if (i == k) {
        L->data[i][i] = 1;
      } else {
        double sum = 0;
        for (int j = 0; j < i; j++) {
          sum += L->data[k][j] * U->data[j][i];
        }
        if (U->data[i][i] == 0) {
          printf("Error: LU decomposition failed. Matrix may be singular.\n");
          free_matrix(L);
          free_matrix(U);
          return NULL;
        }
        L->data[k][i] = (mat->data[k][i] - sum) / U->data[i][i];
      }
    }
  }

  LUDecomposition *result = malloc(sizeof(LUDecomposition));
  result->L = L;
  result->U = U;

  return result;
}

Matrix *solve_linear_system(Matrix *A, Matrix *b) {
  // solves Ax = b using LU decomposition
  LUDecomposition *LU = LUdecompose(A);
  if (LU == NULL) {
    return NULL;
  }

  Matrix *y = solve_lower_triangular(LU->L, b);
  if (y == NULL) {
    free_matrix(LU->L);
    free_matrix(LU->U);
    free(LU);
    return NULL;
  }

  Matrix *x = solve_upper_triangular(LU->U, y);

  free_matrix(LU->L);
  free_matrix(LU->U);
  free(LU);
  free_matrix(y);

  return x;
}

Matrix *invertmat(Matrix *mat){
  int n = mat->rows;

  if (n != mat->cols){
    printf("Error: Matrix has to be square.\n");
    return NULL;
  }

  Matrix *result = malloc_mat(n, n);

  for (int i = 0; i < n; i++){
    // solve Ax=e_i for each i, where e_i are the basis vectors
    // this will give us each column vector of the invert matrix
    Matrix *e = zeros(n, 1);
    e->data[i][0] = 1;
    Matrix *x = solve_linear_system(mat, e);
    for (int k = 0; k < n; k++){
      result->data[k][i] = x->data[k][0]; 
    }
    free_matrix(x);
    free_matrix(e);
  }
  return result;
}

double randfrom(double min, double max){
  double range = (max - min);
  double div = RAND_MAX / range;
  return min + (rand() / div);
}

Matrix *random_vector(int rows){
  assert(rows > 0);

  Matrix *v = malloc_mat(rows, 1);

  if (v == NULL){
    fprintf(stderr, "Error: Memory allocation failed for random vector.\n");
    return NULL;
  }

  for (int i = 0; i < rows; i++){
    v->data[i][0] = randfrom(-1, 1);
  }
  return v;
}

int allclose(const Matrix *v1, const Matrix *v2, double tol){
  // assert inputs are column vectors of same dimension

  assert(v1->rows == v2->rows && v1->cols == 1 && v2->cols == 1);

  int result = 1;
  for (int i = 0; i < v1->rows; i++)
    result = result && (fabs(v1->data[i][0] - v2->data[i][0]) <= tol);
  return result;
}

double dotproduct(const Matrix *v1, const Matrix *v2){
  // assert inputs are column vectors of same dimension
  assert(v1->rows == v2->rows && v1->cols == 1 && v2->cols == 1);

  double res = 0;
  for (int i = 0; i < v1->rows; i++)
    res += v1->data[i][0] * v2->data[i][0];
  return res;
}

double norm2(const Matrix *v){
  // assert input is a column vector
  assert(v->cols == 1);
  return sqrt(dotproduct(v, v));
}

Matrix *scalarmul(const Matrix *mat, double scalar){
  Matrix *result = malloc_mat(mat->rows, mat->cols);
  if (result == NULL){
    fprintf(stderr, "Error: Memory allocation failed.");
    return NULL;
  }

  result->rows = mat->rows;
  result->cols = mat->cols;

  for (int i = 0; i < mat->rows; i++)
    for (int j = 0; j < mat->cols; j++)
      result->data[i][j] = scalar * mat->data[i][j];
  
  return result;
}

void copy_data(Matrix *source, Matrix *dest){
  assert(source->rows == dest->rows && source->cols == dest->cols);

  for (int i = 0; i < source->rows; i++)
    for (int j = 0; j < source->cols; j++)
      dest->data[i][j] = source->data[i][j];
}

Matrix *power_iteration(const Matrix *A, double tol, int max_iter){
  assert(A->rows == A->cols);
  assert(tol > 0 && max_iter > 0);

  Matrix *v = random_vector(A->cols);

  v = scalarmul(v, 1.0 / norm2(v));
  
  Matrix *previous = zeros(A->cols, 1);

  int iter = 0;

  while (iter < max_iter){
    copy_data(v, previous);
    Matrix *temp = matmul(A, v);
    free_matrix(v);
    v = scalarmul(temp, 1.0 / norm2(temp));
    free_matrix(temp);
    if (allclose(v, previous, TOL))
      break;
    ++iter;
  }

  free_matrix(previous);

  return v;
}

int main() {
  srand(time(NULL)); // set seed for rand

  Matrix *sym_matrix = malloc_mat(3, 3);

  sym_matrix->data[0][0] = 2.92;
  sym_matrix->data[0][1] = 0.86;
  sym_matrix->data[0][2] = -1.15;
  sym_matrix->data[1][0] = 0.86;
  sym_matrix->data[1][1] = 6.51;
  sym_matrix->data[1][2] = 3.32;
  sym_matrix->data[2][0] = -1.15;
  sym_matrix->data[2][1] = 3.32;
  sym_matrix->data[2][2] = 4.57;

  Matrix *eigenvector_1 = power_iteration(sym_matrix, TOL, MAX_ITER);

  PRINT_MATRIX(sym_matrix);
  PRINT_MATRIX(eigenvector_1);

  return 0;
}
