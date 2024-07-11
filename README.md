Principal Components Analysis in C, from scratch

src/linalg.c : various linear algebra utils and methods
src/pca.c : computes the PCA of a .csv dataset

compile with: gcc -o pca src/pca.c -lm
usage: ./pca <n_components> <filename>
