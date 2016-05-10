fprintf('First we have a random 16 x 16 positive-definite matrix:\n')

r = randn(16,16);

G = r'*r

pause

fprintf('eigenvalues of G (according to Matlab ): \n')

eig(G)

fprintf('eigenvalues of G (according to ParaLanczos gpu_svd): \n')

[eigs , det1, err1] = gpu_svd(G);

eigs

pause

fprintf('First 4 x 4 (upper-left) submatrix of G:\n ')

G(1:4,1:4)

pause

fprintf('Eigenvalues of first 4 x 4 submatrix according to Matlab: \n')

eig(G(1:4,1:4))

fprintf('Eigenvalues of first 4 x 4 submatrix according to ParaLanczos gpu_subsvd: \n')

[eigs , det1, err1] = gpu_subsvd(G, 0:1:3);

eigs

pause

fprintf('Eigenvalues of lower-right 4 x 4 submatrix according to Matlab: \n')

eig(G(5:8,5:8))

fprintf('Eigenvalues of upper-left and lower-right submatrices of G according to ParaLanczos gpu_multi_subsvd: \n')


[eigs , det1, err1] = gpu_multi_subsvd(G, [0:1:7], 4);

eigs