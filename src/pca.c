#include <stdio.h>
#include <stdlib.h>

// macro that will print a header containing the name of the matrix
// uses the # operator of the C preprocessor
#define PRINT_MATRIX(mat)                                                      \
  printf("Matrix %s:\n", #mat);                                                \
  print_matrix(mat)

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
void free_mat_from_memory(Matrix *mat);
Matrix *identity(int n);
Matrix *diag(int n, double value);
LUDecomposition *LUdecompose(const Matrix *mat);

void print_matrix(const Matrix *mat) {
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      printf("%8.3f ", mat->data[i][j]);
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

void free_mat_from_memory(Matrix *mat) {
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
      free_mat_from_memory(x);
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
      free_mat_from_memory(x);
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
          free_mat_from_memory(L);
          free_mat_from_memory(U);
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

int main() {
  Matrix *A, *B, *b;

  A = diag(3, -1);
  B = diag(3, 2);
  b = ones(3, 1);

  Matrix *C = matmul(A, B);
  Matrix *D = matsum(A, B);

  LUDecomposition *LU = LUdecompose(C);
  Matrix *L = LU->L;
  Matrix *U = LU->U;

  Matrix *y = solve_lower_triangular(L, b);
  Matrix *x = solve_upper_triangular(U, y);

  if (C == NULL || D == NULL)
    return 1;
  PRINT_MATRIX(A);
  PRINT_MATRIX(B);
  PRINT_MATRIX(C);
  PRINT_MATRIX(x);
  return 0;
}
