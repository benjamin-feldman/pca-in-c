#include <stdio.h>
#include <stdlib.h>

// macro that will print a header containing the name of the matrix
// uses the # operator of the C preprocessor
#define PRINT_MATRIX(mat) printf("Matrix %s:\n", #mat); print_matrix(mat)

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

void print_matrix(const Matrix *mat);
Matrix *matmul(const Matrix *A, const Matrix *B);
Matrix *malloc_mat(int rows, int cols);
void free_mat_from_memory(Matrix *mat);
Matrix *identity(int n);
Matrix *diag(int n, double value);

void print_matrix(const Matrix *mat){
    for (int i = 0; i < mat->rows; i++){
        for (int j = 0; j < mat->cols; j++){
            printf("%8.3f ", mat->data[i][j]);
        }
        printf("\n");
    }
}
            
Matrix *malloc_mat(int rows, int cols){
    Matrix *mat = malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = malloc(rows*sizeof(double *));
    for (int i = 0; i < rows; i++) {
        mat->data[i] = malloc(cols * sizeof(double));
    }
    return mat;
}

void free_mat_from_memory(Matrix *mat){
    for (int i = 0; i < mat->rows; i++){
        free(mat->data[i]);
    }
    free(mat->data);
}

Matrix *matmul(const Matrix *A, const Matrix *B){
    // returns a pointer to a matrix, so that it can also return NULL
    // in C, NULL is defined as ((void*)0)
    if (A->cols != B->rows){
        printf("Error: Matrices do not have matching dimensions.\n");
        return NULL;
    }

    Matrix *C = malloc_mat(A->rows, B->cols);

    for (int i = 0; i < A->rows; i++){
        for (int j = 0; j < B->cols; j++){
            double Cij = 0;
            for (int k = 0; k < A->cols; k++){
                Cij += A->data[i][k] * B->data[k][j];
            }
            C->data[i][j] = Cij;
        }
    }
    return C;
}

Matrix *matsum(const Matrix *A, const Matrix *B){
    if (A->rows != B->rows || A->cols != B->cols){
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

Matrix *identity(int n){
    Matrix *m = malloc_mat(n, n);
    for (int i = 0; i < n; i++)
        m->data[i][i] = 1.;
    return m;
}

Matrix *diag(int n, double value){
    Matrix *m = malloc_mat(n, n);
    for (int i = 0; i < n; i++)
        m->data[i][i] = value;
    return m;
}


int main(){
    Matrix *A, *B;

    A = diag(3, -1);
    B = diag(3, 1);

    Matrix *C = matmul(A, B);
    Matrix *D = matsum(A, B);

    if (C == NULL || D == NULL) return 1;
    PRINT_MATRIX(A);
    PRINT_MATRIX(B);
    PRINT_MATRIX(C);
    return 0;
}
