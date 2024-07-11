#include "linalg.h"

Matrix *malloc_matrix(int rows, int cols) {
  Matrix *mat = malloc(sizeof(Matrix));
  if (!mat)
    return NULL;

  mat->rows = rows;
  mat->cols = cols;
  mat->data = malloc(rows * sizeof(double *));
  if (!mat->data) {
    free(mat);
    return NULL;
  }

  for (int i = 0; i < rows; i++) {
    mat->data[i] = calloc(cols, sizeof(double));
    if (!mat->data[i]) {
      for (int j = 0; j < i; j++)
        free(mat->data[j]);
      free(mat->data);
      free(mat);
      return NULL;
    }
  }
  return mat;
}

void free_matrix(Matrix *mat) {
  if (!mat)
    return;
  for (int i = 0; i < mat->rows; i++) {
    free(mat->data[i]);
  }
  free(mat->data);
  free(mat);
}

void matrix_print(const Matrix *mat) {
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      printf("%8.2f ", mat->data[i][j]);
    }
    printf("\n");
  }
}

Matrix *matrix_copy(const Matrix *source) {
  Matrix *dest = malloc_matrix(source->rows, source->cols);
  if (!dest)
    return NULL;

  for (int i = 0; i < source->rows; i++)
    for (int j = 0; j < source->cols; j++)
      dest->data[i][j] = source->data[i][j];
  return dest;
}

Matrix *matrix_identity(int n) {
  Matrix *m = malloc_matrix(n, n);
  if (!m)
    return NULL;
  for (int i = 0; i < n; i++)
    m->data[i][i] = 1.0;
  return m;
}

Matrix *matrix_diagonal(int n, double value) {
  Matrix *m = malloc_matrix(n, n);
  if (!m)
    return NULL;
  for (int i = 0; i < n; i++)
    m->data[i][i] = value;
  return m;
}

Matrix *matrix_ones(int n, int p) {
  Matrix *m = malloc_matrix(n, p);
  if (!m)
    return NULL;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < p; j++)
      m->data[i][j] = 1.0;
  return m;
}

Matrix *matrix_zeros(int n, int p) {
  return malloc_matrix(n, p); // calloc already initializes to zero
}

Matrix *matrix_random_vector(int rows) {
  Matrix *v = malloc_matrix(rows, 1);
  if (!v)
    return NULL;
  for (int i = 0; i < rows; i++) {
    v->data[i][0] = ((double)rand() / RAND_MAX) * 2 - 1; // Range [-1, 1]
  }
  return v;
}

Matrix *matrix_multiply(const Matrix *A, const Matrix *B) {
  if (A->cols != B->rows)
    return NULL;

  Matrix *C = malloc_matrix(A->rows, B->cols);
  if (!C)
    return NULL;

  for (int i = 0; i < A->rows; i++) {
    for (int j = 0; j < B->cols; j++) {
      for (int k = 0; k < A->cols; k++) {
        C->data[i][j] += A->data[i][k] * B->data[k][j];
      }
    }
  }
  return C;
}

Matrix *matrix_add(const Matrix *A, const Matrix *B) {
  if (A->rows != B->rows || A->cols != B->cols)
    return NULL;

  Matrix *C = malloc_matrix(A->rows, A->cols);
  if (!C)
    return NULL;

  for (int i = 0; i < A->rows; i++)
    for (int j = 0; j < A->cols; j++)
      C->data[i][j] = A->data[i][j] + B->data[i][j];
  return C;
}

Matrix *matrix_scalar_multiply(const Matrix *mat, double scalar) {
  Matrix *result = malloc_matrix(mat->rows, mat->cols);
  if (!result)
    return NULL;

  for (int i = 0; i < mat->rows; i++)
    for (int j = 0; j < mat->cols; j++)
      result->data[i][j] = scalar * mat->data[i][j];

  return result;
}

double vector_dot_product(const Matrix *v1, const Matrix *v2) {
  assert(v1->rows == v2->rows && v1->cols == 1 && v2->cols == 1);
  double res = 0;
  for (int i = 0; i < v1->rows; i++)
    res += v1->data[i][0] * v2->data[i][0];
  return res;
}

double vector_norm2(const Matrix *v) {
  assert(v->cols == 1);
  return sqrt(vector_dot_product(v, v));
}

int vector_allclose(const Matrix *v1, const Matrix *v2, double tol) {
  assert(v1->rows == v2->rows && v1->cols == 1 && v2->cols == 1);
  for (int i = 0; i < v1->rows; i++)
    if (fabs(v1->data[i][0] - v2->data[i][0]) > tol)
      return 0;
  return 1;
}

MatrixDecomposition *matrix_LU_decomposition(const Matrix *mat) {
  /* uses the Doolittle algorithm */
  if (mat->rows != mat->cols)
    return NULL;

  int n = mat->rows;
  Matrix *L = malloc_matrix(n, n);
  Matrix *U = malloc_matrix(n, n);
  if (!L || !U) {
    free_matrix(L);
    free_matrix(U);
    return NULL;
  }

  for (int i = 0; i < n; i++) {
    // upper triangular
    for (int k = i; k < n; k++) {
      double sum = 0;
      for (int j = 0; j < i; j++)
        sum += L->data[i][j] * U->data[j][k];
      U->data[i][k] = mat->data[i][k] - sum;
    }

    // lower triangular
    for (int k = i; k < n; k++) {
      if (i == k)
        L->data[i][i] = 1;
      else {
        double sum = 0;
        for (int j = 0; j < i; j++)
          sum += L->data[k][j] * U->data[j][i];
        if (U->data[i][i] == 0) {
          free_matrix(L);
          free_matrix(U);
          return NULL;
        }
        L->data[k][i] = (mat->data[k][i] - sum) / U->data[i][i];
      }
    }
  }

  MatrixDecomposition *result = malloc(sizeof(MatrixDecomposition));
  if (!result) {
    free_matrix(L);
    free_matrix(U);
    return NULL;
  }
  result->left_matrix = L;
  result->right_matrix = U;
  return result;
}

MatrixDecomposition *matrix_QR_decomposition(const Matrix *mat) {
  /* computes the QR decomposition using gram-schmidt */
  assert(mat->rows == mat->cols);
  int n = mat->rows;

  Matrix *Q = malloc_matrix(n, n);
  Matrix *R = malloc_matrix(n, n);
  if (!Q || !R) {
    free_matrix(Q);
    free_matrix(R);
    return NULL;
  }

  for (int i = 0; i < n; i++) {
    Matrix *a = matrix_get_column(mat, i);
    Matrix *u = matrix_copy(a);

    for (int k = 0; k < i; k++) {
      Matrix *ek = matrix_get_column(Q, k);
      double proj = vector_dot_product(a, ek);
      Matrix *proj_vec = matrix_scalar_multiply(ek, proj);
      Matrix *temp = matrix_add(u, matrix_scalar_multiply(proj_vec, -1));

      free_matrix(u);
      u = temp;
      free_matrix(ek);
      free_matrix(proj_vec);
    }

    double norm_u = vector_norm2(u);
    if (norm_u < 1e-10) {
      free_matrix(u);
      u = matrix_zeros(n, 1);
      u->data[i][0] = 1;
    } else {
      Matrix *e = matrix_scalar_multiply(u, 1.0 / norm_u);
      free_matrix(u);
      u = e;
    }

    matrix_set_column(Q, u, i);

    for (int j = i; j < n; j++) {
      Matrix *aj = matrix_get_column(mat, j);
      R->data[i][j] = vector_dot_product(aj, u);
      free_matrix(aj);
    }

    free_matrix(a);
    free_matrix(u);
  }

  MatrixDecomposition *QR = malloc(sizeof(MatrixDecomposition));

  if (!QR) {
    free_matrix(Q);
    free_matrix(R);
    return NULL;
  }

  QR->left_matrix = Q;
  QR->right_matrix = R;
  return QR;
}

Matrix *matrix_solve_linear_system(const Matrix *A, const Matrix *b) {
  MatrixDecomposition *LU = matrix_LU_decomposition(A);
  if (!LU)
    return NULL;

  Matrix *y = matrix_solve_lower_triangular(LU->left_matrix, b);
  if (!y) {
    free_matrix(LU->left_matrix);
    free_matrix(LU->right_matrix);
    free(LU);
    return NULL;
  }

  Matrix *x = matrix_solve_upper_triangular(LU->right_matrix, y);

  free_matrix(LU->left_matrix);
  free_matrix(LU->right_matrix);
  free(LU);
  free_matrix(y);

  return x;
}

Matrix *matrix_inverse(const Matrix *mat) {
  if (mat->rows != mat->cols)
    return NULL;

  int n = mat->rows;
  Matrix *result = malloc_matrix(n, n);
  if (!result)
    return NULL;

  for (int i = 0; i < n; i++) {
    Matrix *e = matrix_zeros(n, 1);
    if (!e) {
      free_matrix(result);
      return NULL;
    }
    e->data[i][0] = 1;

    Matrix *x = matrix_solve_linear_system(mat, e);
    if (!x) {
      free_matrix(result);
      free_matrix(e);
      return NULL;
    }

    for (int k = 0; k < n; k++) {
      result->data[k][i] = x->data[k][0];
    }

    free_matrix(x);
    free_matrix(e);
  }
  return result;
}

Matrix *matrix_get_column(const Matrix *mat, int j) {
  assert(j < mat->cols);
  Matrix *col = malloc_matrix(mat->rows, 1);
  if (!col)
    return NULL;
  for (int i = 0; i < mat->rows; i++)
    col->data[i][0] = mat->data[i][j];
  return col;
}

void matrix_set_column(Matrix *mat, const Matrix *column_vector, int j) {
  assert(column_vector->cols == 1 && column_vector->rows == mat->rows);
  for (int i = 0; i < mat->rows; i++)
    mat->data[i][j] = column_vector->data[i][0];
}

Matrix *matrix_solve_lower_triangular(const Matrix *L, const Matrix *b) {
  int n = b->rows;
  if (n != L->cols)
    return NULL;

  Matrix *x = malloc_matrix(n, 1);
  if (!x)
    return NULL;

  for (int i = 0; i < n; i++) {
    double sum = 0;
    for (int j = 0; j < i; j++)
      sum += L->data[i][j] * x->data[j][0];
    if (L->data[i][i] == 0) {
      free_matrix(x);
      return NULL;
    }
    x->data[i][0] = (b->data[i][0] - sum) / L->data[i][i];
  }
  return x;
}

Matrix *matrix_solve_upper_triangular(const Matrix *U, const Matrix *b) {
  int n = b->rows;
  if (n != U->cols)
    return NULL;

  Matrix *x = malloc_matrix(n, 1);
  if (!x)
    return NULL;

  for (int i = n - 1; i >= 0; i--) {
    double sum = 0;
    for (int j = i + 1; j < n; j++)
      sum += U->data[i][j] * x->data[j][0];
    if (U->data[i][i] == 0) {
      free_matrix(x);
      return NULL;
    }
    x->data[i][0] = (b->data[i][0] - sum) / U->data[i][i];
  }
  return x;
}

Matrix *matrix_power_iteration(const Matrix *A, double tol, int max_iter) {
  assert(A->rows == A->cols);
  assert(tol > 0 && max_iter > 0);

  Matrix *v = matrix_random_vector(A->cols);
  if (!v)
    return NULL;

  Matrix *v_norm = matrix_scalar_multiply(v, 1.0 / vector_norm2(v));
  free_matrix(v);
  v = v_norm;

  Matrix *previous = matrix_zeros(A->cols, 1);
  if (!previous) {
    free_matrix(v);
    return NULL;
  }

  int iter = 0;
  while (iter < max_iter) {
    free_matrix(previous);
    previous = matrix_copy(v);

    Matrix *temp = matrix_multiply(A, v);
    if (!temp) {
      free_matrix(v);
      free_matrix(previous);
      return NULL;
    }

    free_matrix(v);
    v = matrix_scalar_multiply(temp, 1.0 / vector_norm2(temp));
    free_matrix(temp);

    if (vector_allclose(v, previous, TOL))
      break;

    ++iter;
  }

  free_matrix(previous);
  return v;
}

int main() {
  srand(time(NULL));

  // test symmetric matrix
  Matrix *sym_matrix = malloc_matrix(3, 3);
  sym_matrix->data[0][0] = 2.92;
  sym_matrix->data[0][1] = 0.86;
  sym_matrix->data[0][2] = -1.15;
  sym_matrix->data[1][0] = 0.86;
  sym_matrix->data[1][1] = 6.51;
  sym_matrix->data[1][2] = 3.32;
  sym_matrix->data[2][0] = -1.15;
  sym_matrix->data[2][1] = 3.32;
  sym_matrix->data[2][2] = 4.57;

  printf("Symmetric Matrix:\n");
  matrix_print(sym_matrix);

  Matrix *eigenvector_1 = matrix_power_iteration(sym_matrix, TOL, MAX_ITER);
  printf("\nFirst Eigenvector:\n");
  matrix_print(eigenvector_1);

  // test QR decomposition
  Matrix *A = malloc_matrix(3, 3);
  A->data[0][0] = 1;
  A->data[0][1] = 1;
  A->data[0][2] = 0;
  A->data[1][0] = 1;
  A->data[1][1] = 0;
  A->data[1][2] = 1;
  A->data[2][0] = 0;
  A->data[2][1] = 1;
  A->data[2][2] = 1;

  printf("\nMatrix A:\n");
  matrix_print(A);

  MatrixDecomposition *QR = matrix_QR_decomposition(A);
  printf("\nQ Matrix:\n");
  matrix_print(QR->left_matrix);
  printf("\nR Matrix:\n");
  matrix_print(QR->right_matrix);

  // free allocated memory
  free_matrix(sym_matrix);
  free_matrix(eigenvector_1);
  free_matrix(A);
  free_matrix(QR->left_matrix);
  free_matrix(QR->right_matrix);
  free(QR);

  return 0;
}