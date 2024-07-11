# Principal Components Analysis in C, from scratch

- `src/linalg.c`: Various linear algebra utils and methods
- `src/pca.c`: Computes the PCA of a .csv dataset, saves it to a new .csv

Compile with:
`gcc -o pca src/pca.c -lm`

Usage:
`./pca <n_components> <filename>`
